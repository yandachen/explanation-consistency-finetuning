from expl_generation import train_hyperparam_tuning, predict_expl_ans
from score_consistency import generate_simulate_cfs, taskqa_cfs, score_consistency
from consistency_data_augmentation import generate_consistency_data, merge_orig_consistent_data


if __name__ == '__main__':
    # generate EC data
    cfgen_format='list'
    orig_data_path = '../data/strategyqa.json'
    dems_path='../ec_dems/strategyqa.json'
    exp_dir = '../experiments/strategyqa/ec'
    ec_data_path = f'{exp_dir}/ec_data.json'
    generate_consistency_data(data_path=orig_data_path, dems_path=dems_path,
                              out_path=f'{exp_dir}/ec_relqa_data.pkl',
                              qn_type='yn', cfgen_format=cfgen_format, cfans_format='obo', model='llm')
    merge_orig_consistent_data(data_path=orig_data_path,
                               rel_qa_path=f'{exp_dir}/ec_relqa_data.pkl',
                               qn_type='yn', out_data_path=ec_data_path)
    
    # EC training + inference
    qn_type = 'yn'
    expl_type = 'cot'
    train_hyperparam_tuning(model_name='llama2-chat-13B', pretrained_model_name_or_path='meta-llama/Llama-2-13b-chat-hf',
							data_fname=ec_data_path, qn_type=qn_type, expl_type=expl_type, 
                            lrs=[1e-5, 3e-5, 1e-4], num_epochs=10, bsz=8, patience=0, 
                            exp_dir=exp_dir)
    model_dir = f'{exp_dir}/model/'
    predict_expl_ans(model_name='llama2-chat-13B', pretrained_model_name_or_path='meta-llama/Llama-2-13b-chat-hf',
                     load_model_weights_dir=model_dir, data_fname=orig_data_path, qn_type=qn_type, expl_type=expl_type,
                     out_dir=model_dir)
    
    # Score explanation consistency
    cfgen_format = 'list'
    num_samples = 20
    score_consistency_dir = f'{model_dir}/score_consistency/cf_gen_{cfgen_format}_{num_samples}'
    generate_simulate_cfs(qn_type='yn', input_path=orig_data_path, expl_path=f'{model_dir}/greedy_preds.json', dems_path=dems_path,
                          cfgen_format=cfgen_format, cfans_format='obo', num_samples=num_samples,
                          score_consistency_dir=score_consistency_dir)
    taskqa_cfs(model_name='llama2-chat-13B', pretrained_model_name_or_path='meta-llama/Llama-2-13b-chat-hf', 
               load_model_weights_dir=model_dir, score_consistency_dir=score_consistency_dir, qn_type='yn', expl_type='cot')
    score_consistency(score_consistency_dir, qn_type='yn')
    