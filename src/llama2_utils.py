# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import re
from transformers import AutoModelForCausalLM, AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


def convert_llama2_prompt_format(system_prompt, user_prompt, system_response=None):
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if system_response is not None:
        system_response = system_response.strip()
    assert not any([tag in system_prompt for tag in SPECIAL_TAGS])
    assert not any([tag in user_prompt for tag in SPECIAL_TAGS])
    prompt = f"{B_INST} {(B_SYS + system_prompt + E_SYS + user_prompt).strip()} {E_INST}"
    if system_response is None:
        return prompt
    else:
        return prompt + ' ' + system_response


def convert_llama2_multiturn_human_assistant_prompt(prompt):
    prompt = prompt.strip()
    # turns = prompt.split('\n\n')
    pattern = re.compile(r"(Human|Assistant): (.*?)(?=Human: |Assistant: |$)", re.DOTALL)
    matches = pattern.findall(prompt)
    turns = []
    for role, message in matches:
        turns.append({"role": role, "message": message.strip()})
    # check that the turns are alternating between human and assistant
    for turn_idx, turn in enumerate(turns):
        if turn_idx % 2 == 0:
            assert turns[turn_idx]['role'] == 'Human', turns[turn_idx]
        else:
            assert turns[turn_idx]['role'] == 'Assistant', turns[turn_idx]
    # embed into llama prompt
    # llama_prompt = convert_llama2_prompt_format(system_prompt='You are a helpful assistant.', user_prompt=turns[0]['message'])
    llama_prompt = convert_llama2_prompt_format(system_prompt='You are a helpful assistant. You must follow the instruction from the user very carefully. You must answer questions in all turns in exactly the same format. Make sure your answer is formal. Do not use words such as \"sure\", \"great\", etc. Remember, you must answer all turns in exactly the same format (including capitalization, spacing and line breaks), and you must follow the instruction from the user very carefully (especially the instruction from the first turn).', user_prompt=turns[0]['message'])
    # llama_prompt = convert_llama2_prompt_format(system_prompt='You are a helpful assistant. You must follow the instruction from the user very carefully. Make sure your outputs satisfy all requirements specify by the user. Responses that fail to conform to the format required by the user will be rejected.', user_prompt=turns[0]['message'])
    for turn_idx in range(1, len(turns), 2):
        assert (turns[turn_idx]['role'] == 'Assistant') and (turns[turn_idx+1]['role'] == 'Human')
        model_msg = turns[turn_idx]['message']
        user_msg = turns[turn_idx+1]['message']
        # capitalize first letters
        model_msg = model_msg[0].upper() + model_msg[1:]
        user_msg = user_msg[0].upper() + user_msg[1:]
        llama_prompt += f' {model_msg} </s><s>[INST] {user_msg} [/INST]'
    return llama_prompt

