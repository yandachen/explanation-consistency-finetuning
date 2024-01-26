from copy import deepcopy
from load_prompt import PromptLoader
from api_wrapper import ApiWrapper
from model_wrapper import generate_wrapper


class CF_Gen():
    def __init__(self, qn_type, dems_path):
        self.promptloader = PromptLoader(dems_path=dems_path, qn_type=qn_type)
        self.qn_type = qn_type

    def remove_response_prefix(self, text):
        text = text.strip()
        prefixes = ['Assistant: here is my response.', 'here is my response.', 'Here is my response.']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text

    def parse_question(self, text):
        text = self.remove_response_prefix(text).strip()
        lines = text.split('\n')
        if self.qn_type == 'mc':
            if len(lines) < 5:  # one question + 4 choices
                return None
            if lines[0].startswith('Follow-up Question:'):
                lines[0] = lines[0][len('Follow-up Question:'):].strip()
            options = ['A', 'B', 'C', 'D']
            for option_idx in range(4):
                if not lines[option_idx + 1].startswith(f'{options[option_idx]}. '):
                    return None
            return {'question': '\n'.join([line.strip() for line in lines[:5]])}
        elif self.qn_type == 'yn':
            # if len(lines) != 2 or not lines[0].strip().endswith("?"):
            #     return None
            if lines[0].startswith('Follow-up Question:'):
                lines[0] = lines[0][len('Follow-up Question:'):].strip()
            if not lines[0].strip().endswith("?"):
                return None
            question = lines[0].strip()
            return {'question': question}
        else:
            raise NotImplementedError

    def parse_question_list(self, text):
        text = self.remove_response_prefix(text).strip()
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) != 0]
        questions = []
        if self.qn_type == 'mc':
            line_ptr = 0
            for idx in range(len(lines) // 5):
                if lines[line_ptr].startswith(f'{idx+1}.'):
                    lines[line_ptr] = lines[line_ptr][len(
                        f'{idx+1}.'):].strip()
                    question = self.parse_question('\n'.join(lines[line_ptr: line_ptr+5]))
                    if question is not None:
                        questions.append(question)
                    line_ptr += 5
                else:
                    break
        elif self.qn_type == 'yn':
            for line in lines:
                if '.' in line and line[: line.index('.')].strip().isdigit():
                    line = line[line.index('.')+1:].strip()
                if line.endswith('?'):
                    questions.append({'question': line})
        else:
            raise NotImplementedError
        return questions

    def generate_obo(self, model, inputs, expls, num_samples, top_p, use_api, chat_prompt, debug=False):
        assert len(inputs) == len(expls)
        num_examples = len(inputs)
        prompts = [self.promptloader.load_prompt_qg_obo(test_ex={'orig_qn': input['question'], 'orig_qa_tm_expl': expl['explanation']}, chat_prompt=chat_prompt)
                   for input, expl in zip(inputs, expls)]
        if debug:
            print([prompts[0]])
        # repeat the prompts for self.num_samples times
        prompts = [prompt for prompt in prompts for _ in range(num_samples)]
        assert len(prompts) == num_examples * num_samples
        if use_api:
            apiwrapper = ApiWrapper()
            responses = apiwrapper.multiprocess_api(model_name=model, prompts=prompts, num_processes=16 if model == 'gpt3' else 12,
                                                    temperature=1, top_p=top_p, max_tokens=600, stop='\n\n')
        else:
            responses = generate_wrapper(model, prompts, temperature=1, top_p=top_p, max_tokens=600, stop='\n\n')
        assert len(responses) == len(prompts)
        sim_inputs = [self.parse_question(response) for response in responses]
        # group the generated outputs by examples
        assert len(sim_inputs) == num_examples * num_samples
        example_siminputs = []
        for ex_idx in range(num_examples):
            ex_sim_inputs = [sim_input for sim_input in sim_inputs[ex_idx * num_samples: (ex_idx + 1) * num_samples]
                             if sim_input is not None]
            seen_questions = set()
            unique_idxs = []
            for idx in range(len(ex_sim_inputs)):
                qn = ex_sim_inputs[idx]['question']
                if qn not in seen_questions:
                    seen_questions.add(qn)
                    unique_idxs.append(idx)
            ex_sim_inputs = [ex_sim_inputs[idx] for idx in unique_idxs]
            example_siminputs.append(ex_sim_inputs)
        assert len(example_siminputs) == num_examples
        return example_siminputs

    def generate_list(self, model, inputs, expls, num_samples, top_p, use_api, chat_prompt, debug=False):
        assert len(inputs) == len(expls)
        num_examples = len(inputs)
        prompts = [self.promptloader.load_prompt_qg_list(test_ex={'orig_qn': input['question'], 'orig_qa_tm_expl': expl['explanation'], 'num_samples': num_samples}, chat_prompt=chat_prompt)
                   for input, expl in zip(inputs, expls)]
        if debug:
            print([prompts[0]])
        assert len(prompts) == num_examples
        stop_token = {'yn': '\n\n', 'mc': '\n\n\n'}[self.qn_type]
        if use_api:
            apiwrapper = ApiWrapper()
            responses = apiwrapper.multiprocess_api(model_name=model, prompts=prompts, num_processes=16 if model == 'gpt3' else 12,
                                                    temperature=1, top_p=top_p, max_tokens=200 * num_samples, stop=stop_token)
        else:
            responses = generate_wrapper(model, prompts, temperature=1, top_p=top_p, max_tokens=200 * num_samples, stop=stop_token)
        assert len(responses) == num_examples
        example_siminputs = [self.parse_question_list(response) for response in responses]
        return example_siminputs
