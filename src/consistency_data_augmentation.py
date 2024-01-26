import os
import random
import sys
import json
import pickle as pkl
from tqdm import tqdm, trange
from cf_gen import CF_Gen
from cf_ans import CF_Ans


def relevant_qg_qa(cf_gen_model, cf_ans_model, qn_type, inputs, expls, dems_path, top_p, num_samples, cfgen_format, cfans_format,
                   cf_gen_use_api, cf_ans_use_api, cf_ans_post_processing, cf_ans_post_processing_use_api=None, debug=False):
    cf_generator = CF_Gen(qn_type=qn_type, dems_path=dems_path)
    assert len(inputs) == len(expls)
    if cfgen_format == 'list':
        cfs = cf_generator.generate_list(
            cf_gen_model, inputs, expls, num_samples, top_p, cf_gen_use_api, True, debug)
    else:
        assert cfgen_format == 'obo'
        cfs = cf_generator.generate_obo(
            cf_gen_model, inputs, expls, num_samples, top_p, cf_gen_use_api, True, debug)
    cf_answerer = CF_Ans('relevantqa', qn_type=qn_type, dems_path=dems_path)
    assert cfans_format == 'obo'
    cf_preds = cf_answerer.answer_obo(
        cf_ans_model, inputs, expls, cfs, cf_ans_use_api, post_processing=cf_ans_post_processing, debug=debug, post_processing_use_api=cf_ans_post_processing_use_api)
    assert len(cfs) == len(cf_preds) == len(inputs)
    return [{'counterfactuals': cfs[idx], 'simulation_ans': cf_preds[idx]} for idx in range(len(cfs))]


def generate_consistency_data(data_path, dems_path, out_path, qn_type, cfgen_format, cfans_format, model):
    assert model in ['llm', 'llama2-chat-13B']
    if model == 'llm':
        cf_gen_model = ['gpt-4-1106-preview']
        cf_ans_model = 'claude-2'
        cf_gen_use_api, cf_ans_use_api, cf_ans_post_processing_use_api = True, True, True
        num_samples = 10
    else:
        cf_gen_model = 'llama2-chat-13B'
        cf_ans_model = 'llama2-chat-13B'
        cf_gen_use_api, cf_ans_use_api, cf_ans_post_processing_use_api = False, False, False
        num_samples = 20
        
    data = json.load(open(data_path))
    out_dir = out_path[:out_path.rindex('/')]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path):
        split2exidx2rels = pkl.load(open(out_path, 'rb'))
    else:
        split2exidx2rels = {split: {} for split in ['train', 'dev']}
    for split in ['train', 'dev']:
        split_data = data[split]
        exidxs = range(len(split_data))
        exidxs = [exidx for exidx in exidxs if exidx not in split2exidx2rels[split]]
        bsz = 100
        ptr = 0
        for ptr in tqdm(range(0, len(exidxs), bsz)):
            batch_exidxs = exidxs[ptr: ptr + bsz]
            outputs = relevant_qg_qa(cf_gen_model=cf_gen_model, cf_ans_model=cf_ans_model, qn_type=qn_type,
                                     inputs=[split_data[exidx] for exidx in batch_exidxs], expls=[split_data[exidx] for exidx in batch_exidxs],
                                     dems_path=dems_path, top_p=1.0, num_samples=num_samples,
                                     cfgen_format=cfgen_format, cfans_format=cfans_format,
                                     cf_gen_use_api=cf_gen_use_api, cf_ans_use_api=cf_ans_use_api, 
                                     cf_ans_post_processing=True, cf_ans_post_processing_use_api=cf_ans_post_processing_use_api,
                                     debug=False)
            assert len(batch_exidxs) == len(outputs), (len(batch_exidxs), len(outputs))
            for exidx, output in zip(batch_exidxs, outputs):
                split2exidx2rels[split][exidx] = output
            pkl.dump(split2exidx2rels, open(out_path, 'wb'))


def check_explanation_legal(expl):
    expl = expl.lower()
    if ('starter' in expl) or ('qa' in expl) or ('according to the information' in expl) or ('information provided' in expl) or ('\n' in expl):
        return False
    return True


def merge_orig_consistent_data(data_path, rel_qa_path, qn_type, out_data_path):
    split2data = json.load(open(data_path, 'r'))
    split2exidx2rels = pkl.load(open(rel_qa_path, 'rb'))
    split2cadata = {}
    for split in ['train', 'dev']:
        assert len(split2data[split]) == len(split2exidx2rels[split])
        cadata = []
        for exidx in range(len(split2exidx2rels[split])):
            cfs, sim_answers = split2exidx2rels[split][exidx]['counterfactuals'], split2exidx2rels[split][exidx]['simulation_ans']
            cadata.append(split2data[split][exidx])
            for cfidx in range(len(cfs)):
                if sim_answers[cfidx]['answer'] in {'mc': ['A', 'B', 'C', 'D'], 'yn': ['yes', 'no']}[qn_type]:
                    explanation = sim_answers[cfidx]['explanation']
                    if check_explanation_legal(explanation):
                        cadata.append({'question': cfs[cfidx]['question'],
                                       'explanation': explanation,
                                       'answer': sim_answers[cfidx]['answer']})
        split2cadata[split] = cadata
    split2cadata['test'] = split2data['test']
    for split in split2cadata:
        print(split, len(split2cadata[split]))
    out_data_dir = out_data_path[:out_data_path.rindex('/')]
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir, exist_ok=True)
    assert not os.path.exists(out_data_path)
    json.dump(split2cadata, open(out_data_path, 'w'), indent=4)

