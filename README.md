# Towards Consistent Natural Language Explanations via Explanation-Consistency Finetuning

This is the implementation of the paper [Towards Consistent Natural Language Explanations via Explanation-Consistency Finetuning](https://arxiv.org/abs/2401.13986). 
## Table of Contents
* [Overview](#overview)
* [Requirements](#requirements)
* [Code Structure](#code-structure)
* [Demo](#demo)
* [How to Cite](#citation)


## Overview
Large language models (LLMs) often generate convincing, fluent explanations. However, their explanations are often *inconsistent* on different inputs. For example, an LLM may generate the explanation "*all birds can fly*" when answering the question "*Can sparrows fly?*", but answer "*no*" to the question "*Can penguins fly?*".
Explanations should be consistent on different inputs so that they should allow a human to simulate the LLM's decision process on multiple inputs based on the LLM's explanation for a single input.

We propose ***Explanation-consistency Finetuning*** (EC Finetuning), a method that adapts LLMs so that they generate more consistent natural-language explanations.
EC Finetuning involves  finetuning an LLM on synthetic data that is carefully constructed to contain consistent explanations.
Across a variety of question-answering datasets, EC Finetuning improves the consistency of natural-language explanations, and generalizes to datasets which were not seen during finetuning.

You could find more details of this work in our [paper](https://arxiv.org/abs/2401.13986).

## Requirements

To run our code, please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

## Code Structure
Code is within the `src/` directory:
- `cf_gen.py` generates counterfactuals relevant to an explanation.

- `cf_ans.py` generates answers to counterfactuals consistent with the model's explanations on other inputs.

- `consistency_data_augmentation.py` generates EC training data.

- `expl_generation.py` finetunes CoT QA systems on (input, explanation, output) tuples.

- `score_consistency.py` scores the consistency of the model's explanations.

We include the demonstration examples we use to generate EC training data in `ec_dems/`.


## Demo
As an example, we provide a code file `src/main.py` that runs the entire pipeline (generate EC training data, finetune CoT models on EC data, score the consistency of explanations).

This file can be run simply with `cd src && python main.py` 

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at `yanda.chen@cs.columbia.edu`.

## Citation

```bibtex
@misc{chen2024consistent,
      title={Towards Consistent Natural-Language Explanations via Explanation-Consistency Finetuning}, 
      author={Yanda Chen and Chandan Singh and Xiaodong Liu and Simiao Zuo and Bin Yu and He He and Jianfeng Gao},
      year={2024},
      eprint={2401.13986},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```