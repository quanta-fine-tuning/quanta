# (NeurIPS 2024) QuanTA: Efficient High-Rank Fine-Tuning of LLMs with Quantum-Informed Tensor Adaptation
Official implementation of **QuanTA**: Efficient High-Rank Fine-Tuning of LLMs with **Quan**tum-Informed **T**ensor **A**daptation (https://arxiv.org/abs/2406.00132)

![Example Image](figures/quanta_illustration.jpg)

### To cite our paper
```
@inproceedings{
    chen2024quanta,
    author = {Chen, Zhuo and Dangovski, Rumen and Loh, Charlotte and Dugan, Owen and Luo, Di and Solja\v{c}i\'{c}, Marin},
    booktitle = {Advances in Neural Information Processing Systems},
    doi = {10.52202/079017-2928},
    editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
    pages = {92210--92245},
    publisher = {Curran Associates, Inc.},
    title = {{QuanTA}: Efficient High-Rank Fine-Tuning of LLMs with Quantum-Informed Tensor Adaptation},
    url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/a7c17115db36193f6b83b71b0fe1d416-Paper-Conference.pdf},
    volume = {37},
    year = {2024}
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


