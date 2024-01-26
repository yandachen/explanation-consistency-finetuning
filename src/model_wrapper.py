from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
import os
import sys
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from llama2_utils import convert_llama2_prompt_format, convert_llama2_multiturn_human_assistant_prompt
import gc


class CLM_wrapper:
    def __init__(self, model_name, pretrained_model_name_or_path, load_model_weight_dir=None, model_parallel=False):
        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                       use_auth_token='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')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        # setup model
        device_map = None if model_parallel is False else 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16,
                                                            use_auth_token='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',
                                                            device_map=device_map)
        if device_map is None:
            self.model.cuda()
        if load_model_weight_dir is not None:
            self.model.load_state_dict(torch.load(load_model_weight_dir, map_location='cpu'))
        self.model_name = model_name

    def collate_fn(self, dataset):
        # dataset is a list of (input, output), input=str, output=str
        # sanity check
        for ex_idx in range(len(dataset)):
            input, output = dataset[ex_idx]
            # assert '\n' not in input and '\n' not in output
            assert '\n' not in output
            if self.model_name.startswith('gpt'):
                dataset[ex_idx] = (input.strip(), ' ' + output.strip())
            else:
                dataset[ex_idx] = (input.strip() + ' ', output.strip())

        input_output_texts = [ex[0] + ex[1] for ex in dataset]
        input_dict = self.tokenizer(input_output_texts, padding=True, return_tensors="pt")
        num_examples = len(input_output_texts)

        # build output labels (pad tokens and input tokens are masked with 100 in the labels variable)
        labels = input_dict['input_ids'].detach().clone()
        for example_idx in range(num_examples):
            encoded_output = self.tokenizer(dataset[example_idx][1])['input_ids']
            if self.model_name.startswith('alpaca') or self.model_name.startswith('llama2'):
                assert encoded_output[0] == self.tokenizer.bos_token_id
                encoded_output = encoded_output[1:]
            assert labels[example_idx][-len(encoded_output):].tolist() == encoded_output
            labels[example_idx][:-len(encoded_output)] = -100
        input_dict['labels'] = labels
        return input_dict

    def collate_fn_nolabel(self, inputs):
        # inputs is a list of input (str)
        inputs = [input.strip() for input in inputs]
        input_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")
        return input_dict

    def collate_fn_pad(self, input_ids):
        # input_ids is List[list] of ints
        max_length = np.max([len(ex_input_ids) for ex_input_ids in input_ids])
        padded_inputs = np.full((len(input_ids), max_length), self.tokenizer.pad_token_id)
        attention_mask = np.zeros((len(input_ids), max_length))
        for ex_idx in range(len(input_ids)):
            padded_inputs[ex_idx][-len(input_ids[ex_idx]):] = input_ids[ex_idx]
            attention_mask[ex_idx][-len(input_ids[ex_idx]):] = 1
        return {'input_ids': torch.tensor(padded_inputs).int(), 'attention_mask': torch.tensor(attention_mask).int()}

    def train(self, train_data, dev_data, lr, num_epochs, bsz, num_grad_acc, patience, output_dir, 
              shuffle=True, use_lr_scheduler=False):
        assert not os.path.exists(output_dir), output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.model.gradient_checkpointing_enable()

        with open(f'{output_dir}/train.log', 'a') as f:
            f.write(str({'lr': lr, 'num_epochs': num_epochs, 'bsz': bsz, 'num_grad_acc': num_grad_acc, 'patience': patience,
                         'shuffle': shuffle, 'use_lr_scheduler': use_lr_scheduler}) + '\n')
            
        # load model
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # training
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        optimizer.zero_grad()
        optimal_dev_loss = np.inf
        optimal_epoch = 0
        if use_lr_scheduler:
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, 
                                                           num_training_steps=len(train_data) // (bsz * num_grad_acc) * num_epochs)
        
        optimal_model = None
        for epoch in trange(num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = []
            # prepare data
            train_dataloader = DataLoader(train_data, shuffle=shuffle, batch_size=bsz, collate_fn=self.collate_fn)
            # train
            batch_idx = 0
            for batch in tqdm(train_dataloader):
            # for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                batch = {k: v.cuda() for k, v in batch.items()}
                if 'token_type_ids' in batch:
                    del batch['token_type_ids']
                loss = self.model(**batch).loss / num_grad_acc
                loss.backward()
                train_loss.append(loss.item() * num_grad_acc)
                if (batch_idx + 1) % num_grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if use_lr_scheduler:
                        lr_scheduler.step()
                batch_idx += 1
            train_loss = np.mean(train_loss)
            # evaluate on dev
            dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=bsz, collate_fn=self.collate_fn)
            dev_loss = []
            for batch in dev_dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                if 'token_type_ids' in batch:
                    del batch['token_type_ids']
                loss = self.model(**batch).loss.item()
                dev_loss += [loss] * len(batch['input_ids'])
            assert len(dev_loss) == len(dev_data)
            dev_loss = np.mean(dev_loss)

            if dev_loss < optimal_dev_loss:
                optimal_epoch = epoch
                optimal_dev_loss = dev_loss
                optimal_model = deepcopy(self.model.state_dict())

            epoch_end_time = time.time()
            with open(f'{output_dir}/train.log', 'a') as f:
                f.write(
                    f'Epoch {epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f} ({(epoch_end_time - epoch_start_time)//60} min)\n')

            if epoch - optimal_epoch > patience:
                torch.save(optimal_model, f'{output_dir}/model.pkl')
                return optimal_dev_loss
        torch.save(optimal_model, f'{output_dir}/model.pkl')
        return optimal_dev_loss


    def predict(self, inputs, bsz, eos_token_id, do_sample, num_beams=None, top_p=None, num_return_sequences=1, max_new_tokens=600):
        if do_sample:
            assert (top_p is not None) and (num_beams is None)
        else:
            assert (num_beams is not None) and (top_p is None)
        if num_return_sequences > 1:
            assert do_sample
        # inputs is a list of input (str)
        self.model.eval()
        outputs = []
        dataloader = DataLoader(inputs, batch_size=bsz, collate_fn=self.collate_fn_nolabel)
        for batch in tqdm(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            if 'token_type_ids' in batch:
                del batch['token_type_ids']
            with torch.no_grad():
                if do_sample: # sampling
                    input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, num_return_sequences=num_return_sequences,
                                                        early_stopping=True, eos_token_id=eos_token_id)
                else: # greedy/beam search
                    input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                                        early_stopping=True, eos_token_id=eos_token_id)
            assert len(input_outputs) == len(batch['input_ids']) * num_return_sequences
            # duplicate batch['input_ids'] by num_return_sequences number of times (to match input_outputs)
            batch_input_ids = torch.stack([batch['input_ids'][ex_idx] for ex_idx in range(len(batch['input_ids'])) for _ in range(num_return_sequences)])
            assert len(batch_input_ids) == len(input_outputs)
            assert torch.all(input_outputs[:, : batch_input_ids.shape[1]] == batch_input_ids)
            batch_outputs = input_outputs[:, batch_input_ids.shape[1]:].cpu().tolist()
            for output in batch_outputs:
                if eos_token_id not in output:
                    outputs.append(output)
                else:
                    outputs.append(output[: output.index(eos_token_id)])
        output_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert len(output_texts) == len(inputs) * num_return_sequences
        output_texts = [text.strip() for text in output_texts]
        if num_return_sequences == 1:
            return output_texts
        else:
            assert len(output_texts) % num_return_sequences == 0
            output_texts = [output_texts[pos: pos + num_return_sequences] for pos in range(0, len(output_texts), num_return_sequences)]
            assert len(output_texts) == len(inputs)
            return output_texts
        
    
    def encode(self, input_ids, bsz, device, extract, layers, token_positions):
        # input_ids is List[List] (unpadded)
        # return: 
        assert extract in ['hidden_states', 'attention', 'attention_outputs']
        assert len(input_ids) == len(token_positions)
        self.model.eval()
        self.model.to(device)
        reps = []
        dataloader = DataLoader(input_ids, batch_size=bsz, collate_fn=self.collate_fn_pad)
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # calculate the number of pad tokens (offsets) for each example, so as to locate token_positions
            attention_mask = batch['attention_mask']
            num_examples = attention_mask.shape[0]
            batch_num_pads = torch.sum(attention_mask == 0, dim=1).cpu().numpy()
            batch_token_positions = token_positions[len(reps): len(reps) + num_examples]
            # incorporate the offset from padding
            for ex_idx in range(num_examples):
                for pos_idx in range(len(batch_token_positions[ex_idx])):
                    batch_token_positions[ex_idx][pos_idx] += batch_num_pads[ex_idx]
            with torch.no_grad():
                if extract == 'hidden_states':
                    out = self.model.forward(**batch, output_hidden_states=True)['hidden_states']
                    for ex_idx in range(num_examples):
                        ex_rep = []
                        for layer in layers:
                            ex_rep.append(out[layer+1][ex_idx][batch_token_positions[ex_idx]].cpu().float().numpy())
                        reps.append(np.array(ex_rep))
                elif extract == 'attention':
                    out = self.model.forward(**batch, output_attentions=True)['attentions']
                    for ex_idx in range(num_examples):
                        ex_rep = []
                        for layer in layers:
                            ex_rep.append(out[layer][ex_idx][:, batch_token_positions[ex_idx], :][:, :, batch_token_positions[ex_idx]].cpu().float().numpy())
                        reps.append(ex_rep)
                elif extract == 'attention_outputs':
                    out = self.model.forward(**batch, output_attention_outputs=True)['attn_outputs'] # (num_layers, bsz, q_len, num_heads, head_dim)
                    for ex_idx in range(num_examples):
                        ex_rep = [] # (num_layers, num_heads, token_positions, head_dim)
                        for layer in layers:
                            layer_rep = []
                            for attention_head in range(len(out[0][0][0])):
                                layer_rep.append(out[layer][ex_idx][batch_token_positions[ex_idx]][:, attention_head].cpu().float().numpy())
                            ex_rep.append(np.array(layer_rep))
                        reps.append(np.array(ex_rep)) # (num_exs, num_layers, num_heads, token_positions, head_dim)
        assert len(reps) == len(input_ids)
        return reps
    

def generate_wrapper(model_name, prompts, temperature, top_p, max_tokens, stop=None):
    if model_name == 'llama2-chat-13B':
        mw = CLM_wrapper(model_name='llama2-chat-13B', pretrained_model_name_or_path = 'meta-llama/Llama-2-13b-chat-hf', model_parallel=True)
        # embed prompt with llama2 chat format
        # prompts = [convert_llama2_prompt_format('You are a helpful assistant.', prompt) for prompt in prompts]
        prompts = [convert_llama2_multiturn_human_assistant_prompt(prompt) for prompt in prompts]
        bsz = 8
        if temperature == 0:
            out = mw.predict(prompts, bsz=bsz, eos_token_id=None, do_sample=False, num_beams=1, max_new_tokens=max_tokens)
        else:
            out = mw.predict(prompts, bsz=bsz, eos_token_id=None, do_sample=True, top_p=top_p, max_new_tokens=max_tokens)
        del mw
        torch.cuda.empty_cache()
        gc.collect()
        return out
