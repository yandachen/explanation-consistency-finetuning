import sys
from llama2_utils import convert_llama2_prompt_format
import json
import numpy as np
from model_wrapper import CLM_wrapper
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
import pickle as pkl
import os
import shutil


# cot/posthoc prompts
def embed_prompt(qn_type, chat_prompt, expl_type, question, answer=None, explanation=None, eos_token=None):
    assert qn_type in ['mc', 'yn']
    if not chat_prompt:
        input = f"{question} Answer:"
        if answer is None:
            return input
        assert answer is not None and explanation is not None and eos_token is not None
        if expl_type == 'cot':
            output = f"{explanation} So the answer is {answer}. {eos_token}"
        elif expl_type == 'posthoc':
            output = f"{answer}. Explanation: {explanation} {eos_token}"
        else:
            raise NotImplementedError
        return input, output
    else:
        user_prompt = question
        options = {'mc': 'A/B/C/D', 'yn': 'yes/no'}[qn_type]
        if expl_type == 'cot':
            system_prompt = f'I will ask you a question. First explain your reasoning and then output the answer ({options}).'
        elif expl_type == 'posthoc':
            system_prompt = f'I will ask you a question. First answer the question ({options}) and then explain your reasoning.'
        else:
            raise NotImplementedError
        # return input/input&output
        input = convert_llama2_prompt_format(system_prompt, user_prompt)
        if answer is None:
            return input
        elif expl_type == 'cot':
            output = f"{explanation} So the answer is {answer}. {eos_token}"
        elif expl_type == 'posthoc':
            output = f"{answer}. Explanation: {explanation} {eos_token}"
        else:
            raise NotImplementedError
        return input, output


def extract_answer_from_output(qn_type, expl_type, text):
    assert qn_type in ['mc', 'yn']
    options = {'mc': ['A', 'B', 'C', 'D'], 'yn': ['yes', 'no']}[qn_type]
    text = text.strip()
    if expl_type == 'cot':
        answer_is = [
            f'So the answer is {option}' in text for option in options]
        if sum(answer_is) != 1:
            answer = 'neither'
            explanation = text
        else:
            answer = options[answer_is.index(1)]
            explanation = text[:text.index(
                f'So the answer is {answer}')].strip()
    elif expl_type == 'posthoc':
        answer_is = [text.startswith(option) for option in options]
        if sum(answer_is) != 1:
            answer = 'neither'
        else:
            answer = options[answer_is.index(1)]
        if 'Explanation:' in text:
            explanation = text[text.index(
                'Explanation:') + len('Explanation:'):].strip()
        else:
            explanation = text.strip()
    else:
        raise NotImplementedError
    return {'answer': answer, 'explanation': explanation}


# train model
def train_model(model_name, pretrained_model_name_or_path,
                data, qn_type, expl_type,
                model_parallel, lr, num_epochs, bsz, num_grad_acc, patience, output_dir, shuffle=True, use_lr_scheduler=False):
    # load data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, use_auth_token='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')
    eos_token = tokenizer.eos_token
    chat_prompt = 'llama' in model_name and 'chat' in model_name
    assert (type(data) == list) == (type(qn_type) == list)
    if type(data) != list:  # single-task learning
        train_data = [embed_prompt(qn_type, chat_prompt, expl_type, ex['question'],
                                   ex['answer'], ex['explanation'], eos_token) for ex in data['train']]
        dev_data = [embed_prompt(qn_type, chat_prompt, expl_type, ex['question'],
                                 ex['answer'], ex['explanation'], eos_token) for ex in data['dev']]
    else:  # multi-task learning
        assert len(data) == len(qn_type)
        train_data = [embed_prompt(task_qn_type, chat_prompt, expl_type, ex['question'], ex['answer'], ex['explanation'], eos_token)
                      for task_data, task_qn_type in zip(data, qn_type) for ex in task_data['train']]
        dev_data = [embed_prompt(task_qn_type, chat_prompt, expl_type, ex['question'], ex['answer'], ex['explanation'], eos_token)
                    for task_data, task_qn_type in zip(data, qn_type) for ex in task_data['dev']]
    # train + eval
    clm_wrapper = CLM_wrapper(
        model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path, model_parallel=model_parallel)
    dev_loss = clm_wrapper.train(train_data=train_data, dev_data=dev_data,
                                 lr=lr, num_epochs=num_epochs, bsz=bsz, num_grad_acc=num_grad_acc, patience=patience, output_dir=output_dir,
                                 shuffle=shuffle, use_lr_scheduler=use_lr_scheduler)
    return dev_loss

# test model


def predict_expl_ans(model_name, pretrained_model_name_or_path, load_model_weights_dir,
                     data_fname, qn_type, expl_type, out_dir=None,
                     model_parallel=True, bsz=16, do_sample=False, num_beams=1, top_p=None, num_return_sequences=1):
    # load data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, use_auth_token='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')
    eos_token_id = tokenizer.eos_token_id
    chat_prompt = 'llama' in model_name and 'chat' in model_name
    clm_wrapper = CLM_wrapper(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                              load_model_weight_dir=f'{load_model_weights_dir}/model.pkl', model_parallel=model_parallel)
    if type(data_fname) == str:
        data = json.load(open(data_fname))
        data = data['test']
    elif type(data_fname) == list:
        data = data_fname
    else:
        raise NotImplementedError
    questions = [ex['question'] for ex in data]
    eval_inputs = [embed_prompt(qn_type=qn_type, chat_prompt=chat_prompt,
                                expl_type=expl_type, question=question) for question in questions]
    output_texts = clm_wrapper.predict(inputs=eval_inputs, eos_token_id=eos_token_id, bsz=bsz,
                                       do_sample=do_sample, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences)
    if num_return_sequences == 1:
        preds = [extract_answer_from_output(
            qn_type, expl_type, text) for text in output_texts]
    else:
        preds = [[extract_answer_from_output(
            qn_type, expl_type, text) for text in ex_texts] for ex_texts in output_texts]
    assert len(preds) == len(questions)
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if do_sample is False:
            assert num_beams == 1
            json.dump(preds, open(
                f'{out_dir}/greedy_preds.json', 'w'), indent=4)
        else:
            json.dump(preds, open(
                f'{out_dir}/sampling_preds_topp{top_p}.json'), indent=4)
    return preds


def train_hyperparam_tuning(model_name, pretrained_model_name_or_path,
                            data_fname, qn_type, expl_type, lrs, num_epochs, bsz, patience, exp_dir,
                            effective_bsz=32):
    # load data
    if type(data_fname) == str:
        data = json.load(open(data_fname))
    elif type(data_fname) == list:
        data = [json.load(open(fname)) for fname in data_fname]
        assert (type(qn_type) == list) and (len(data_fname) == len(qn_type))
    num_grad_acc = effective_bsz // bsz
    dev_losses = []
    for lr in lrs:
        dev_loss = train_model(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                               data=data, qn_type=qn_type, expl_type=expl_type,
                               model_parallel=True, lr=lr, num_epochs=num_epochs, bsz=bsz, num_grad_acc=num_grad_acc, patience=patience,
                               output_dir=f'{exp_dir}/lr{lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/', use_lr_scheduler=True)
        dev_losses.append(dev_loss)
    # choose the learning rate with lowest dev loss, change the file directory without the hyperparamters name; remove the other files
    optimal_lr = lrs[np.argmin(dev_losses)]
    optimal_output_dir = f'{exp_dir}/lr{optimal_lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/'
    optimal_output_dir_new = f'{exp_dir}/model/'
    os.rename(optimal_output_dir, optimal_output_dir_new)
    # remove checkpoints of suboptimal hyperparameters
    os.mkdir(f'{exp_dir}/hyperparameter_tuning')
    for lr in lrs:
        if lr != optimal_lr:
            os.remove(
                f'{exp_dir}/lr{lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/model.pkl')
            shutil.move(f'{exp_dir}/lr{lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/',
                        f'{exp_dir}/hyperparameter_tuning')
