import re
from load_prompt import PromptLoader
from model_wrapper import generate_wrapper
from api_wrapper import ApiWrapper
import pickle as pkl


class CF_Ans():
    def __init__(self, mode, qn_type, dems_path):
        assert mode in ['simqa', 'relevantqa', 'taskqa']
        self.mode = mode
        self.promptloader = PromptLoader(dems_path=dems_path, qn_type=qn_type)
        self.qn_type = qn_type
        
    def remove_response_prefix(self, text):
        text = text.strip()
        prefixes = ['Assistant: here is my response.', 'here is my response.', 'Here is my response.']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    def parse_ans(self, text):
        text = self.remove_response_prefix(text).strip()
        options = {'mc': ['unknown', 'A', 'B', 'C', 'D'], 'yn': ['unknown', 'yes', 'no']}[self.qn_type]
        answer_match = [text.endswith(f'So the answer is {option}.') for option in options]
        if sum(answer_match) != 1:
            answer = 'neither'
            explanation = text
        else:
            assert answer_match.index(1) >= 0
            answer = options[answer_match.index(1)]
            explanation = text[:-len(f'So the answer is {answer}.')].strip() # need this for RelevantQA
        return {'explanation': explanation, 'answer': answer}

    def answer_obo(self, model, orig_inputs, expls, sim_inputs_list, use_api, post_processing, post_processing_use_api=None, debug=False):
        assert self.mode in ['simqa', 'relevantqa', 'taskqa']
        if self.mode != 'taskqa':
            assert len(orig_inputs) == len(expls) == len(sim_inputs_list)
            num_examples = len(orig_inputs)
            if self.mode == 'simqa':
                load_prompt_f = self.promptloader.load_prompt_simqa_obo
            elif self.mode == 'relevantqa':
                load_prompt_f = self.promptloader.load_prompt_relevantqa_obo
            else:
                raise NotImplementedError
            prompts = [load_prompt_f(test_ex={'orig_qn': orig_input['question'], 'orig_qa_tm_expl': expl['explanation'], 'sim_qn': sim_input['question']})
             for orig_input, expl, sim_inputs in zip(orig_inputs, expls, sim_inputs_list)
             for sim_input in sim_inputs]
        else:
            assert (orig_inputs is None) and (expls is None)
            num_examples = len(sim_inputs_list)
            load_prompt_f = self.promptloader.load_prompt_taskqa_obo
            prompts = [load_prompt_f(test_ex={'question': sim_input['question']}) for sim_inputs in sim_inputs_list for sim_input in sim_inputs]
        if debug:
            print([prompts[0]])
        # deduplicate the prompts before calling the API to save time
        deduplicated_prompts = list(set(prompts))
        if use_api:
            apiwrapper = ApiWrapper()
            pred_expls = apiwrapper.multiprocess_api(model_name=model, prompts=deduplicated_prompts, num_processes=10, temperature=0, max_tokens=300, stop='\n')
        else:
            pred_expls = generate_wrapper(model, deduplicated_prompts, temperature=0, top_p=0, max_tokens=300, stop='\n')
        assert len(pred_expls) == len(deduplicated_prompts)
        # if model == 'claude', postprocess with gpt-4
        if post_processing:
            if post_processing_use_api:
                pred_expls = self.postprocess_answers_api(pred_expls)
            else:
                pred_expls = self.postprocess_answers_llama(model, pred_expls)
        # add duplicate prompts back
        prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
        pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
        assert len(pred_expls) == len(prompts)
        # extract answers
        preds = []
        for pred_expl in pred_expls:
            preds.append(self.parse_ans(pred_expl))
        # regroup preds according to examples (multiple simulation questions correspond to each original question)
        assert len(preds) == len(prompts)
        example_preds = []
        cur = 0
        for ex_idx in range(num_examples):
            example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
            cur += len(sim_inputs_list[ex_idx])
        assert cur == len(preds)
        return example_preds

    def is_legal(self, response):
        if self.qn_type == 'mc':
            options = ['A', 'B', 'C', 'D']
        elif self.qn_type == 'yn':
            options = ['yes', 'no']
        if self.mode != 'taskqa':
            options.append('unknown')
        for option in options:
            if response.endswith(f'So the answer is {option}.'):
                return True
        return False

    def postprocess_answers_api(self, answers):
        if self.qn_type == 'mc':
            option_string = '"So the answer is A.", "So the answer is B.", "So the answer is C.", "So the answer is D."'
        elif self.qn_type == 'yn':
            option_string = '"So the answer is yes.", "So the answer is no."'
        else:
            raise NotImplementedError
        if self.mode == 'relevantqa':
            answers = [self.remove_response_prefix(answer.strip()) for answer in answers]
            instruction = f'Rewrite the following answer so that 1) The answer must end exactly with {option_string}, or "So the answer is unknown.", 2) If the answer is not unknown, the answer can use the provided information, but cannot directly reference it or use phrases like "according to" or "as stated in". If the answer is unknown, then it\'s okay if it directly references the provided information. 3) Do not drop any information.'
            # only postprocess those simulatable answers
            answers_need_postprocessing = [answer for answer in answers if 'is unknown' not in answer]
            prompts = [instruction + '\n\n' + answer for answer in answers_need_postprocessing]
            apiwrapper = ApiWrapper()
            postprocessing_outputs = apiwrapper.multiprocess_api(model_name=['gpt-4-1106-preview'], prompts=prompts, num_processes=12, temperature=0, max_tokens=300)
            assert len(postprocessing_outputs) == len(answers_need_postprocessing)
            answer2pp_output = {answer: output.strip() for answer, output in zip(answers_need_postprocessing, postprocessing_outputs)}
            cleaned_answers = []
            for answer in answers:
                if answer in answer2pp_output:
                    cleaned_answers.append(answer2pp_output[answer])
                else:
                    assert 'is unknown' in answer
                    if not answer.endswith('So the answer is unknown.'):
                        answer += 'So the answer is unknown.'
                    cleaned_answers.append(answer)
            return cleaned_answers
        elif self.mode in ['simqa', 'taskqa']:
            # only run postprocessing on bad formats
            answers = [self.remove_response_prefix(answer.strip()) for answer in answers]
            answers_need_postprocessing = [answer for answer in answers if not self.is_legal(answer)]
            if self.mode == 'simqa':
                instruction = f'Rewrite the following answer so that the answer must end exactly with {option_string}, or "So the answer is unknown."'
            elif self.mode == 'taskqa':
                instruction = f'Rewrite the following answer so that the answer must end exactly with {option_string}'
            prompts = [instruction + '\n\n' + self.remove_response_prefix(answer.strip()) for answer in answers_need_postprocessing]
            apiwrapper = ApiWrapper()
            postprocessing_outputs = apiwrapper.multiprocess_api(model_name=['gpt-4'], prompts=prompts, num_processes=12, temperature=0, max_tokens=300)
            assert len(postprocessing_outputs) == len(answers_need_postprocessing)
            answer2pp_output = {answer: output.strip() for answer, output in zip(answers_need_postprocessing, postprocessing_outputs)}
            cleaned_answers = [answer if answer not in answer2pp_output else answer2pp_output[answer] for answer in answers]
            assert len(cleaned_answers) == len(answers)
            return cleaned_answers
        else:
            raise NotImplementedError

    
    def postprocess_answers_llama(self, model, answers):
        def rewrite_simulatable_answers(answers):
            answers = [self.remove_response_prefix(answer.strip()).strip() for answer in answers]
            dems = [
                    {'input': 'The starter QA states that Mumbai is located in India, and India is a country located in South Asia. Therefore, the answer to the follow-up question is yes.',
                    'output': 'Mumbai is located in India, and India is a country located in South Asia. So the answer is yes.'},
                    {'input': 'The starter QA directly helps answer the follow-up question. The starter QA states that flying fish are found in the epipelagic zone, which is the top layer of the ocean. Therefore, the answer to the follow-up question is yes.',
                    'output': 'Flying fish are found in the epipelagic zone, which is the top layer of the ocean. So the answer is yes.'},
                    {'input': 'The starter QA states that divers can reach depths of around 1000 feet with a dry dive suit. This implies that a person can breathe normally while wearing a dry dive suit, as they would be able to reach such depths without any issues. So the answer is yes.',
                    'output': 'Divers can reach depths of around 1000 feet with a dry dive suit. Therefore a person can breathe normally while wearing a dry dive suit, as they would be able to reach such depths without any issues. So the answer is yes.'},
                    {'input': 'The starter QA states that divers can reach depths of around 1000 feet with scuba gear. However, the depth of 3800 feet is beyond the limit of what is safely reachable with scuba gear. Therefore, the answer is no.',
                    'output': 'Divers can reach depths of around 1000 feet with scuba gear. However, the depth of 3800 feet is beyond the limit of what is safely reachable with scuba gear. So the answer is no.'},
                    ]
            instruction = "Human: Begin your revised statement by stating the facts in a standalone manner without referring to the original source of information. Avoid using phrases that indicate the source of your information, such as \"according to,\" \"as stated in,\" or \"the starter QA.\" Retain all the key details from the original statement to ensure no information is lost in your revised answer. Conclude your revised answer with the phrase \"So the answer is yes/no\" to clearly indicate your final assertion. This must be the last sentence of your response.\n\nHere are some examples of how to apply these rules:\n\n"
            for dem_idx, dem in enumerate(dems):
                input, output = dem['input'], dem['output']
                instruction += f'Example {dem_idx+1}:\nInput: {input}\nOutput: {output}\n\n'
            instruction += "By following the given instructions, each output retains the essential information from the input without referencing the original source and ends with a definitive conclusion in the exact format of \"So the answer is yes/no.\"\nNow it is your turn. Remember, when you rewrite the provided statement:\n1. Maintain all essential details without citing the original source.\n2. Conclude with the phrase \"So the answer is yes/no.\"\nYour response should follow this format:\n1. Start with \"Output:\".\n2. The final sentence MUST be exactly \"So the answer is yes/no.\". Paraphrases (e.g., \"Therefore the answer is yes.\") are NOT accepted.\n3. Avoid repetition of the concluding phrase, e.g., \"Therefore the answer is yes. So the answer is yes.\"\n4. Do not insert line breaks.\nRemember, the final sentence of your answer MUST be EXACTLY \"So the answer is yes/no.\"."
            prompts = [instruction + 'Input: ' + answer for answer in answers]
            rewritten_answers = generate_wrapper(model_name=model, prompts=prompts, temperature=0, top_p=0, max_tokens=300, stop='\n\n\n')
            assert len(rewritten_answers) == len(answers)
            return rewritten_answers     
        
        def postprocess_rewritten_simulatable_answers(answer):
            answer = answer.strip()
            for option in ['yes', 'no']:
                for replace_end_text in [f'Therefore, the answer is {option}.', f'Therefore the answer is {option}.', f'So, the answer is {option}.', f'so the answer is {option}.', f'so, the answer is {option}.']:
                    answer = answer.replace(replace_end_text, f'So the answer is {option}.')
            for option in ['yes', 'no']:
                for delete_text in [f'Therefore, the answer is {option}.', f'Therefore the answer is {option}.', f'Therefore, the answer to the follow-up question is {option}.', f'Therefore the answer to the follow-up question is {option}.']:
                    answer = answer.replace(delete_text, '')
            for option in ['yes', 'no']:
                if answer.count(f'So the answer is {option}.') > 1:
                    answer = answer.replace(f'So the answer is {option}.', '', answer.count(f'So the answer is {option}.')-1)
            answer = answer.replace('\n', ' ')
            answer = re.sub(' +', ' ', answer) # clean up consecutive spaces
            return answer
        
        def check_legal(answer):
            if '\n' in answer:
                return False
            banned_words = ['the provided', 'the provided information', 'based on', 'Based on', 'information provided in', 'as stated in', 'starter QA', 'starter qa', 'initial example', 'initial explanation']
            for word in banned_words:
                if word in answer:
                    return False
            if (not answer.endswith('So the answer is yes.')) and (not answer.endswith('So the answer is no.')):
                return False
            return True

        assert self.mode == 'relevantqa' # GPT-4 is used for evaluation
        answers = [self.remove_response_prefix(answer.strip()) for answer in answers]
        # unknown answers - do not rewrite, if the answer doesn't end with "So the answer is unknown.", append this to the answer.
        # simulatable answers - 1) rewrite, 2) if ends with "Therefore, the answer is yes/no." or "Therefore the answer is yes/no.", replace that sentence with "So the answer is yes/no.", 
        # 3) if has "Therefore, the answer is yes/no." or "Therefore the answer is yes/no.", remove that sentence.
        # 4) clean up consecutive spaces.
        # 5) finally check legal: 1) banned ngram: provided, according to, as stated in, starter QA, starter qa, 2) ends with So the answer is yes/no., 3) no line breaks
        answer2pp_answer = {}
        to_rewrite_answers = []
        options = ['yes', 'no', 'unknown']
        for answer in answers:
            answer_match = [f'is {option}.' in answer for option in options]
            if sum(answer_match) == 1:
                if answer_match.index(1) in [0, 1]: # simulatable, rewrite
                    to_rewrite_answers.append(answer)
                else:
                    assert answer_match.index(1) == 2 # unsimulatable
                    answer2pp_answer[answer] = answer + 'So the answer is unknown.'
        # rewrite simulatable answers
        rewritten_answers = rewrite_simulatable_answers(to_rewrite_answers)
        rewritten_answers = [postprocess_rewritten_simulatable_answers(answer) for answer in rewritten_answers]
        assert len(to_rewrite_answers) == len(rewritten_answers)
        for to_rewrite_ans, rewritten_ans in zip(to_rewrite_answers, rewritten_answers):
            if check_legal(rewritten_ans):
                answer2pp_answer[to_rewrite_ans] = rewritten_ans
        # flat out
        pp_answers = [answer2pp_answer[answer] if answer in answer2pp_answer else 'illegal' for answer in answers]
        assert len(answers) == len(pp_answers)
        return pp_answers

