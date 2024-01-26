import json


class PromptLoader():
	def __init__(self, dems_path, qn_type):
		assert qn_type in ['mc', 'yn']
		self.qn_type = qn_type
		self.dem_exs = json.load(open(dems_path))
		self.sanity_check_dems()

	def sanity_check_dems(self):
		# assert set(self.dem_exs.keys()) == set(['simulatable', 'unsimulatable', 'qa_dem_order'])
		assert len(self.dem_exs['qa_dem_order']) == len(self.dem_exs['simulatable']) + len(self.dem_exs['unsimulatable'])
		options = {'mc': ['A', 'B', 'C', 'D'], 'yn': ['yes', 'no']}[self.qn_type]
		for ex in self.dem_exs['simulatable']:
			ans = [ex['sim_qa_expl'].endswith(f'So the answer is {answer}.') for answer in options]
			assert sum(ans) == 1
			ans = [ex['relevant_qa_expl'].endswith(f'So the answer is {answer}.') for answer in options]
			assert sum(ans) == 1
			assert 'starter qa' not in ex['relevant_qa_expl'].lower()
		for ex in self.dem_exs['unsimulatable']:
			assert ex['sim_qa_expl'].endswith('So the answer is unknown.')
			assert ex['relevant_qa_expl'].endswith('So the answer is unknown.')

	def check_input_type(self, test_ex, key2type):
		assert set(test_ex.keys()) == set(key2type.keys())
		for key in test_ex:
			assert type(test_ex[key]) == key2type[key]

	def load_prompt_qg_obo(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'orig_qn': str, 'orig_qa_tm_expl': str})
		# prompt
		instruction = "\n\nHuman: In the questions below, you will be asked to read a starter question and its answer. After that you will be asked to write a follow-up question that can be answered based on the starter QA, and write your answer to the follow-up question based on the starter QA. Your follow-up question should be self-contained even without the starter question.\n\nAssistant: here is my response. okay."
		template_no_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}"
		template_with_label = template_no_label + "\n\nAssistant: here is my response. Follow-up Question: {sim_qn}\nAnswer to the Follow-up Question: {sim_qa_expl}"
		# compile
		prompt = instruction
		for dem in self.dem_exs['simulatable']:
			prompt += template_with_label.format(**dem)
		prompt = prompt + template_no_label.format(**test_ex)
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt

	def load_prompt_qg_list(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'orig_qn': str, 'orig_qa_tm_expl': str, 'num_samples': int})
		# prompt
		instruction = "\n\nHuman: In the questions below, you will be asked to read a starter question and its answer. After that you will be asked to write a follow-up question that can be answered based on the starter QA, and write your answer to the follow-up question based on the starter QA. Your follow-up question should be self-contained even without the starter question.\n\nAssistant: here is my response. okay."
		template_with_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}\n\nAssistant: here is my response. Follow-up Question: {sim_qn}\nAnswer to the Follow-up Question: {sim_qa_expl}"
		qn_format = {'mc': '4-choice multiple-choice', 'yn': 'yes or no'}[self.qn_type]
		template_no_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}\n\nWrite {num_samples} follow-up {qn_format} questions. Start each question by \"1. \", \"2. \", etc. Skip writing answers to your follow-up questions. Remember, your task is to write follow-up questions that can be answered based on the starter QA. Your follow-up questions should be self-contained even without the starter question."
		# compile
		prompt = instruction
		for dem in self.dem_exs['simulatable']:
			prompt += template_with_label.format(**dem)
		prompt = prompt + template_no_label.format(**test_ex, qn_format=qn_format)
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt

	def load_prompt_simqa_obo(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'orig_qn': str, 'orig_qa_tm_expl': str, 'sim_qn': str})
		# prompt
		assert set(test_ex.keys()) == {'orig_qn', 'orig_qa_tm_expl', 'sim_qn'}
		options = {'mc': 'A/B/C/D', 'yn': 'yes/no'}[self.qn_type]
		instruction = f"\n\nHuman: In the questions below, you will be asked to read a starter question and its answer. After that you will be asked to read a follow-up question and judge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question and end your answer with \"So the answer is {options}.\". Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\n\nAssistant: here is my response. okay."
		template_no_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}\nFollow-up Question: {sim_qn}\nJudge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question and end your answer with \"So the answer is {options}.\". Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\nAnswer to the Follow-up Question:"
		template_with_label = template_no_label + "\n\nAssistant: here is my response. {sim_qa_expl}"
		# compile
		prompt = instruction
		dem_exs = self.dem_exs['simulatable'] + self.dem_exs['unsimulatable']
		for dem in [dem_exs[idx] for idx in self.dem_exs['qa_dem_order']]:
			prompt += template_with_label.format(**dem, options=options)
		prompt = prompt + template_no_label.format(**test_ex, options=options)
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt

	def load_prompt_simqa_list(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'orig_qn': str, 'orig_qa_tm_expl': str, 'sim_qns': list})
		for qn in test_ex['sim_qns']:
			assert type(qn) == str
		options = {'mc': 'A/B/C/D', 'yn': 'yes/no'}[self.qn_type]
		# prompt
		instruction = f"\n\nHuman: In the questions below, you will be asked to read a starter question and its answer. After that you will be asked to read a follow-up question and judge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question and end your answer with \"So the answer is {options}.\". Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\n\nAssistant: here is my response. okay."
		template_with_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}\nFollow-up Question: {sim_qn}\nJudge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question and end your answer with \"So the answer is {options}.\". Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\nAnswer to the Follow-up Question:\n\nAssistant: here is my response. {sim_qa_expl}"
		# compile
		prompt = instruction
		dem_exs = self.dem_exs['simulatable'] + self.dem_exs['unsimulatable']
		for dem in [dem_exs[idx] for idx in self.dem_exs['qa_dem_order']]:
			prompt += template_with_label.format(**dem, options=options)
		prompt += "\n\nYour turn:"
		for cf_idx, cf in enumerate(test_ex['sim_qns']):
			prompt += f"\n\n{cf_idx+1}. Human: Starter Question: {test_ex['orig_qn']}\nAnswer to the Starter Question: {test_ex['orig_qa_tm_expl']}\nFollow-up Question: {cf}"
		prompt += f"\n\nFor each of the follow-up question above, judge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question and end your answer with \"So the answer is {options}.\". Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong. Start your answer to each follow-up question with \"1.\", \"2.\", etc.\nAnswers to the above Follow-up Questions:"
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt
	
	def load_prompt_relevantqa_obo(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'orig_qn': str, 'orig_qa_tm_expl': str, 'sim_qn': str})
		options = {'mc': 'A/B/C/D', 'yn': 'yes/no'}[self.qn_type]
		# prompt
		instruction = f"\n\nHuman: In the questions below, you will be asked to read a starter question and its answer. After that you will be asked to read a follow-up question and judge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question, explain your reasoning as clearly and as detailed as possible using all relevant information in the starter QA, end your answer with \"So the answer is {options}.\", and do NOT explicitly mention \"the starter QA\" or \"According to the starter QA\" in your answer. Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\n\nAssistant: here is my response. okay."
		template_no_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}\nFollow-up Question: {sim_qn}\nJudge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question, explain your reasoning as clearly and as detailed as possible using all relevant information in the starter QA, end your answer with \"So the answer is {options}.\", and do NOT explicitly mention \"the starter QA\" or \"According to the starter QA\" in your answer. Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\nAnswer to the Follow-up Question:"
		template_with_label = template_no_label + "\n\nAssistant: here is my response. {sim_qa_expl}"
		# compile
		prompt = instruction
		dem_exs = self.dem_exs['simulatable'] + self.dem_exs['unsimulatable']
		for dem in [dem_exs[idx] for idx in self.dem_exs['qa_dem_order']]:
			# replace "sim_qa_expl" with "relevant_qa_expl"
			dem_copy = {key: dem[key] for key in dem}
			dem_copy["sim_qa_expl"] = dem_copy["relevant_qa_expl"]
			prompt += template_with_label.format(**dem_copy, options=options)
		prompt = prompt + template_no_label.format(**test_ex, options=options)
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt

	def load_prompt_relevantqa_list(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'orig_qn': str, 'orig_qa_tm_expl': str, 'sim_qns': list})
		for qn in test_ex['sim_qns']:
			assert type(qn) == str
		options = {'mc': 'A/B/C/D', 'yn': 'yes/no'}[self.qn_type]
		# prompt
		instruction = f"\n\nHuman: In the questions below, you will be asked to read a starter question and its answer. After that you will be asked to read a follow-up question and judge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question, explain your reasoning as clearly and as detailed as possible using all relevant information in the starter QA, end your answer with \"So the answer is {options}.\", and do NOT explicitly mention \"the starter QA\" or \"According to the starter QA\" in your answer. Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\n\nAssistant: here is my response. okay."
		template_with_label = "\n\nHuman: Starter Question: {orig_qn}\nAnswer to the Starter Question: {orig_qa_tm_expl}\nFollow-up Question: {sim_qn}\nJudge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question, explain your reasoning as clearly and as detailed as possible using all relevant information in the starter QA, end your answer with \"So the answer is {options}.\", and do NOT explicitly mention \"the starter QA\" or \"According to the starter QA\" in your answer. Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong.\nAnswer to the Follow-up Question:\n\nAssistant: here is my response. {sim_qa_expl}"
		# compile
		prompt = instruction
		dem_exs = self.dem_exs['simulatable'] + self.dem_exs['unsimulatable']
		for dem in [dem_exs[idx] for idx in self.dem_exs['qa_dem_order']]:
			# replace "sim_qa_expl" with "relevant_qa_expl"
			dem_copy = {key: dem[key] for key in dem}
			dem_copy["sim_qa_expl"] = dem_copy["relevant_qa_expl"]
			prompt += template_with_label.format(**dem_copy, options=options)
		prompt += "\n\nYour turn:"
		for cf_idx, cf in enumerate(test_ex['sim_qns']):
			prompt += f"\n\n{cf_idx+1}. Human: Starter Question: {test_ex['orig_qn']}\nAnswer to the Starter Question: {test_ex['orig_qa_tm_expl']}\nFollow-up Question: {cf}"
		prompt += f"\n\nFor each of the follow-up question above, judge whether the starter QA directly helps choosing a single answer for the follow-up question. If not, end your answer with \"So the answer is unknown.\". If yes, use the starter QA to answer the follow-up question, explain your reasoning as clearly and as detailed as possible using all relevant information in the starter QA, end your answer with \"So the answer is {options}.\", and do NOT explicitly mention \"the starter QA\" or \"According to the starter QA\" in your answer. Stick to the starter QA when you answer the follow-up question, even if the reasoning or claims in the starter QA are wrong. Start your answer to each follow-up question with \"1.\", \"2.\", etc.\nAnswers to the above Follow-up Questions:"
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt
	
	def load_prompt_taskqa_obo(self, test_ex, chat_prompt=True):
		# check ex format
		self.check_input_type(test_ex, {'question': str})
		options = {'mc': 'A/B/C/D', 'yn': 'yes/no'}[self.qn_type]
		# prompt
		instruction = f"\n\nHuman: In the questions below, you will be asked to first generate a trace of reasoning and then end with your final answer exactly with \"So the answer is ...\" {options}. Strictly follow the example format below and do not say anything else.\n\nAssistant: here is my response. okay."
		template_no_label = "\n\nHuman: Q: Yes or no: {question}\nA:"
		template_with_label = "\n\nHuman: Q: Yes or no: {question}\nA:\n\nAssistant: here is my response. {explanation}"
		# compile
		prompt = instruction
		dem_exs = self.dem_exs['taskqa']
		for dem in dem_exs:
			prompt += template_with_label.format(**dem, options=options)
		prompt = prompt + template_no_label.format(**test_ex, options=options)
		if not chat_prompt:
			prompt += '\n\nAssistant: here is my response.'
		assert '{options}' not in prompt
		return prompt