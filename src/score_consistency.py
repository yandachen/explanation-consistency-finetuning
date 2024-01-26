import numpy as np
import os
import json
import pickle as pkl
from cf_gen import CF_Gen
from cf_ans import CF_Ans
from expl_generation import predict_expl_ans
from datetime import datetime


def simulate_qg_qa(cf_gen_model, cf_ans_model, qn_type, inputs, expls, dems_path, num_samples,
                   cfgen_format, cfans_format, cfgen_model_is_chat,
                   cf_gen_use_api, cf_ans_use_api, cf_ans_post_processing, cf_ans_post_processing_use_api=None,
                   debug=False):
    cf_generator = CF_Gen(qn_type=qn_type, dems_path=dems_path)
    assert len(inputs) == len(expls)
    if cfgen_format == 'list':
        cfs = cf_generator.generate_list(
            cf_gen_model, inputs, expls, num_samples, 1.0, cf_gen_use_api, cfgen_model_is_chat, debug)
    else:
        assert cfgen_format == 'obo'
        cfs = cf_generator.generate_obo(
            cf_gen_model, inputs, expls, num_samples, 1.0, cf_gen_use_api, cfgen_model_is_chat, debug)
    cf_answerer = CF_Ans('simqa', qn_type=qn_type, dems_path=dems_path)
    assert cfans_format == 'obo'
    cf_preds = cf_answerer.answer_obo(
        cf_ans_model, inputs, expls, cfs, cf_ans_use_api, debug=debug, post_processing=cf_ans_post_processing, post_processing_use_api=cf_ans_post_processing_use_api)
    assert len(cfs) == len(cf_preds) == len(inputs)
    print([len(ex_cfs) for ex_cfs in cfs])
    print([len([cf_pred for cf_pred in ex_cf_preds if cf_pred['answer'] in ['A', 'B', 'C', 'D', 'yes', 'no']]) for ex_cf_preds in cf_preds])
    return [{'counterfactuals': cfs[idx], 'simulation_ans': cf_preds[idx]} for idx in range(len(cfs))]


def generate_simulate_cfs(qn_type, input_path, expl_path, dems_path, cfgen_format, cfans_format, num_samples, score_consistency_dir):
    if not os.path.exists(score_consistency_dir):
        os.makedirs(score_consistency_dir, exist_ok=True)
    out_file = f"{score_consistency_dir}/cf_data.pkl"
    # logging
    now = datetime.now()
    now_str = now.strftime("%Y/%m/%d %H:%M:%S")
    if os.path.exists(out_file):
        print(f'{now_str} {out_file} already exists... Skipping...', flush=True)
        return
    else:
        print(f'{now_str} Producing {out_file}...', flush=True)
    inputs = json.load(open(input_path))['test']
    expls = json.load(open(expl_path))
    assert len(expls) == len(inputs)
    exidxs = range(len(inputs))
    outputs = simulate_qg_qa(cf_gen_model='gpt-4-1106-preview',
                             cf_ans_model='claude-2', qn_type=qn_type,
                             inputs=[inputs[exidx] for exidx in exidxs],
                             expls=[expls[exidx] for exidx in exidxs],
                             dems_path=dems_path, num_samples=num_samples,
                             cfgen_format=cfgen_format, cfans_format=cfans_format, cfgen_model_is_chat=False,
                             cf_gen_use_api=True, cf_ans_use_api=True, 
                             cf_ans_post_processing=True, cf_ans_post_processing_use_api=True
                             )
    preds = {}
    for exidx, output in zip(exidxs, outputs):
        preds[exidx] = {key: output[key] for key in ['counterfactuals', 'simulation_ans']}
    pkl.dump(preds, open(out_file, 'wb'))


def taskqa_cfs(model_name, pretrained_model_name_or_path, load_model_weights_dir, score_consistency_dir, qn_type, expl_type):
    exidx2simulation_data = pkl.load(
        open(f'{score_consistency_dir}/cf_data.pkl', 'rb'))
    exidx2expl_counterfactuals = {
        exidx: exidx2simulation_data[exidx]['counterfactuals'] for exidx in exidx2simulation_data}
    outfile = f'{score_consistency_dir}/model_ans.pkl'
    if os.path.exists(outfile):
        exidx2model_ans = pkl.load(open(outfile, 'rb'))
    else:
        exidx2model_ans = {}
    exidxs = sorted([exidx for exidx in exidx2simulation_data if exidx not in exidx2model_ans])
    # flatten the counterfactuals
    flattened_counterfactuals = [
        cf for exidx in exidxs for cf in exidx2expl_counterfactuals[exidx]]
    preds = predict_expl_ans(model_name, pretrained_model_name_or_path, load_model_weights_dir,
                             flattened_counterfactuals, qn_type, expl_type, bsz=8)
    # regroup counterfactual preds to expls and then to examples
    pos = 0
    for exidx in exidxs:
        num_cfs = len(exidx2expl_counterfactuals[exidx])
        exidx2model_ans[exidx] = preds[pos: pos + num_cfs]
        pos += num_cfs
    assert pos == len(preds)
    pkl.dump(exidx2model_ans, open(outfile, 'wb'))


def score_consistency(score_consistency_dir, qn_type):
    exidx2model_ans = pkl.load(
        open(f'{score_consistency_dir}/model_ans.pkl', 'rb'))
    exidx2cf_data = pkl.load(
        open(f'{score_consistency_dir}/cf_data.pkl', 'rb'))
    assert set(exidx2model_ans.keys()) == set(exidx2cf_data.keys())
    exidx2consistencys = {}
    for exidx in exidx2cf_data:
        cfs_simulation_answer = [cf['answer']
                                 for cf in exidx2cf_data[exidx]['simulation_ans']]
        cfs_model_answer = [cf['answer'] for cf in exidx2model_ans[exidx]]
        consistencys = calculate_consistency_example(
            qn_type, cfs_model_answer, cfs_simulation_answer)
        exidx2consistencys[exidx] = consistencys
    pkl.dump(exidx2consistencys, open(
        f'{score_consistency_dir}/consistencys.pkl', 'wb'))


def calculate_consistency_example(qn_type, taskqa_preds, simqa_preds):
    assert qn_type in ['yn', 'mc']
    options = {'yn': ['yes', 'no'], 'mc': ['A', 'B', 'C', 'D']}[qn_type]
    assert len(taskqa_preds) == len(simqa_preds)
    consistencys = []
    for qn_idx in range(len(simqa_preds)):
        if simqa_preds[qn_idx] in options:
            consistencys.append(simqa_preds[qn_idx] == taskqa_preds[qn_idx])
        else:
            consistencys.append(np.nan)
    return consistencys

