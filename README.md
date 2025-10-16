<div align="center">
<h1> 🐍 The Hidden Attention of Mamba Models 🐍 (ACL 2025)</h1>

Ameen Ali<sup>1</sup> \*,Itamar Zimerman<sup>1</sup> \* and Lior Wolf<sup>1</sup>
<br>
ameenali023@gmail.com, itamarzimm@gmail.com, liorwolf@gmail.com 
<br>
<sup>1</sup>  Tel Aviv University 
(\*) equal contribution



</div>

## Official PyTorch Implementation of "The Hidden Attention of Mamba Models"

The Mamba layer offers an efficient state space model (SSM) that is highly effective in modeling multiple domains including long-range sequences and images. SSMs are viewed as dual models, in which one trains in parallel on the entire sequence using convolutions, and deploys in an autoregressive manner. We add a third view and show that such models can be viewed as attention-driven models. This new perspective enables us to compare the underlying mechanisms to that of the self-attention layers in transformers and allows us to peer inside the inner workings of the Mamba model with explainability methods. 
<br>
You can access the paper through : <a href="https://arxiv.org/pdf/2403.01590.pdf">The Hidden Attention of Mamba Models</a>

<div align="center">
<img src="assets/2.png" alt="Left Image" align="center"   width="1000" height="400">
</div>

## Set Up Environment

- Python 3.10.13

  - `conda create -n your_env_name python=3.10.13`
- Activate Env
  - `conda activate your_env_name`
- CUDA TOOLKIT 11.8
  - `conda install nvidia/label/cuda-11.8.0::cuda-toolkit`
- torch 2.1.1 + cu118
  - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- Requirements: vim_requirements.txt
  - `pip install -r vim/vim_requirements.txt`

- Install jupyter
  - `pip install jupyter`
  
- Install ``causal_conv1d`` and ``mamba`` from *<b>our source</b>*
  - `cd causal-conv1d`
  - `pip install --editable .`
  - `cd ..`
  - `pip install --editable mamba-1p1p1`
  
  


## Pre-Trained Weights

We have used the official weights provided by [Vim](https://github.com/hustvl/Vim), which can be downloaded from here:

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [Vim-tiny](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   76.1   | 93.0 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-tiny<sup>+</sup>](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   78.3   | 94.2 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   80.5   | 95.1 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-small<sup>+</sup>](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   81.6   | 95.4 | https://huggingface.co/hustvl/Vim-small-midclstok |

**Notes:**
- <b> In all of our experiments, we have worked with [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok).</b>

## Vision-Mamba Explainability Notebook:
<div align="center">
<img src="assets/xai_gradmethod.jpg" alt="Left Image" align="center"  width="1000" height="300">
</div>
<br>
Follow the instructions in <b>vim/vmamba_xai.ipynb</b> notebook, in order to apply a single-image inference for the 3 introduced methods in the paper.
<br>
<div align="center">
<img src="assets/notebook.png" alt="Left Image" align="center"  width="600" height="600">
</div>

## To-Do
For the segmentation experiment, please check out our [follow-up work](https://github.com/Itamarzimm/UnifiedImplicitAttnRepr/tree/main).
<br>
<ul>
    <strike><li><input type="checkbox" id="task1" checked disabled><label for="task1">XAI - Single Image Inference Notebook</label></li></strike>
    <strike><li><input type="checkbox" id="task2" checked disabled><label for="task2">XAI - Segmentation Experimnts</label></li></strike>
</ul>

## Citation
if you find our work useful, please consider citing us:
```latex
@inproceedings{ali-etal-2025-hidden,
    title="The Hidden Attention of Mamba Models",
    author = "Ali, Ameen Ali  and
      Zimerman, Itamar  and
      Wolf, Lior",
    editor="Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle="Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month=jul,
    year="2025",
    address="Vienna, Austria",
    publisher="Association for Computational Linguistics",
    url="https://aclanthology.org/2025.acl-long.76/",
    doi="10.18653/v1/2025.acl-long.76",
    pages="1516--1534",
    ISBN="979-8-89176-251-0",
    abstract="The Mamba layer offers an efficient selective state-space model (SSM) that is highly effective in modeling multiple domains, includingNLP, long-range sequence processing, and computer vision. Selective SSMs are viewed as dual models, in which one trains in parallel on the entire sequence via an IO-aware parallel scan, and deploys in an autoregressive manner. We add a third view and show that such models can be viewed as attention-driven models. This new perspective enables us to empirically and theoretically compare the underlying mechanisms to that of the attention in transformers and allows us to peer inside the inner workings of the Mamba model with explainability methods. Our code is publicly available."
}
```
## Acknowledgement
This repository is heavily based on [Vim](https://github.com/hustvl/Vim), [Mamba](https://github.com/state-spaces/mamba) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability). Thanks for their wonderful works.
