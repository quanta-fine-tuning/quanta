# (NeurIPS 2024) QuanTA: Efficient High-Rank Fine-Tuning of LLMs with Quantum-Informed Tensor Adaptation
Official implementation of **QuanTA**: Efficient High-Rank Fine-Tuning of LLMs with **Quan**tum-Informed **T**ensor **A**daptation (https://arxiv.org/abs/2406.00132)

![Example Image](figures/quanta_illustration.jpg)

### To cite our paper
```
@inproceedings{
      chen2024quanta,
      title={Quan{TA}: Efficient High-Rank Fine-Tuning of {LLM}s with Quantum-Informed Tensor Adaptation},
      author={Zhuo Chen and Rumen Dangovski and Charlotte Loh and Owen M Dugan and Di Luo and Marin Soljacic},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=EfpZNpkrm2}
}
```

## Quickstart

 ```bash
git clone https://github.com/quanta-fine-tuning/quanta.git
 ```
 ```bash
cd quanta/quanta/
 ```
 ```bash
pip install -e .
 ```
 ```bash
pip install wandb datasets accelerate sentencepiece opt_einsum
 ```
 ```bash
cd ../run/
 ```
 ```bash
sh run.sh
 ```
##### Note:
`numpy` may need to be downgraded to `1.26.4`


