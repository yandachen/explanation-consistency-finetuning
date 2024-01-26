import time
import openai
from tqdm import trange, tqdm
import multiprocessing
from functools import partial
import math
from datetime import datetime
import os


model_queue = multiprocessing.Queue()
openai.api_key = 'XXXXXXX'
openai.organization = 'XXXXXX'

class ApiWrapper():
	def __init__(self):
		pass

	def reset_model_queue(self, num_processes, modelnames):
		while not model_queue.empty():
			model_queue.get()
		for _ in range(num_processes):
			for modelname in modelnames:
				model_queue.put(modelname)

	def multiprocess_api(self, model_name, prompts, num_processes, **kwargs):
		assert type(model_name) in [str, list]
		if type(model_name) == str:
			modelnames = [model_name]
		else:
			modelnames = model_name

		# starting a batch
		assert 'model' not in kwargs and 'engine' not in kwargs
		bsz = 1
		manager = multiprocessing.Manager()
		return_dict = manager.dict()
		p = multiprocessing.Pool(processes=num_processes * len(modelnames))
		assert 'prompt' not in kwargs and 'prompts' not in kwargs # "prompt" is assigned by multiprocessing
		assert 'message' not in kwargs and 'messages' not in kwargs
		
		self.reset_model_queue(num_processes, modelnames)
		
		prompts = [prompt.lstrip() for prompt in prompts]
		call_api = partial(self.singleprocess_api, **kwargs)
		api_call_args = []
		for batch_idx in range(math.ceil(len(prompts) / bsz)):
			batch_prompts = prompts[batch_idx * bsz: (batch_idx + 1) * bsz]
			api_call_args.append((batch_idx, return_dict, batch_prompts))
		pbar = tqdm(p.imap(call_api, api_call_args), total=len(api_call_args))
		for _ in pbar:
			pass
		p.close()
		p.join()

		responses = []
		if 'n' in kwargs:
			n = kwargs['n']
		else:
			n = 1
		for batch_idx in range(len(api_call_args)):
			response = return_dict[batch_idx]
			num_prompts_in_batch = len(api_call_args[batch_idx][-1])
			assert num_prompts_in_batch * n == len(response.choices)
			batch_responses = [''] * (num_prompts_in_batch * n)
			for choice in response.choices:
				batch_responses[choice.index] = choice.message.content.strip()
			if n > 1:
				# group batch_responses
				batch_responses = [batch_responses[exidx * n: (exidx + 1) * n] for exidx in range(num_prompts_in_batch)]
				assert len(batch_responses) == num_prompts_in_batch
				for ex_responses in batch_responses:
					assert len(ex_responses) == n
			assert len(batch_responses) == num_prompts_in_batch
			responses += batch_responses
		assert len(responses) == len(prompts)
		return responses


	def singleprocess_api(self, batchidx_returndict_batchprompts, **kwargs):
		assert 'prompt' not in kwargs and 'prompts' not in kwargs
		assert 'messages' not in kwargs and 'message' not in kwargs
		assert 'model' not in kwargs and 'engine' not in kwargs
		success = False
		response = None
		batch_idx, return_dict, batch_prompts = batchidx_returndict_batchprompts	

		while not success: # launch requests until success
			modelname = model_queue.get()
			try:
				kwargs['model'] = modelname
				assert len(batch_prompts) == 1
				prompt = batch_prompts[0]
				messages = [{"role": "system", "content": "You are a helpful assistant that performs tasks following user instructions."},
							{"role": "user", "content": prompt.strip()}]
				response = openai.ChatCompletion.create(messages=messages, **kwargs)
				success = True
			except Exception as e:
				print(e)
				success = False
				time.sleep(2)
			model_queue.put(modelname)
		assert success  # execute until success
		return_dict[batch_idx] = response
		