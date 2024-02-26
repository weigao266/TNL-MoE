---
license: other
language:
- en
- zh
pipeline_tag: text-generation
tags:
- ' TransNormerLLM'
---

<div align="center">
<h1>
  TransNormerLLM -- A Faster and Better LLM
</h1>
</div>

<p align="center">
💻 <a href="https://github.com/OpenNLPLab/TransnormerLLM" target="_blank">GitHub </a> • 💬 <a href="https://discord.gg/W4Vr7AKW" target="_blank">Discord</a> • 💬 <a href="./images/contact_me_qr.png" target="_blank">Wechat</a> 
</p>


# Table of Contents 

- [Introduction](#introduction)
- [Released Weights](#released-weights)
- [Benchmark Results](#benchmark-results)
  - [General Domain](#general-domain)
    - [Model Results](#model-results)
- [Inference and Deployment](#inference-and-deployment)
  - [Dependency Installation](#dependency-installation)
  - [Notice](#notice)
  - [Python Code Inference](#python-code-inference)
    - [Demonstration of Base Model Inference](#demonstration-of-base-model-inference)
- [Fine-tuning the Model](#fine-tuning-the-model)
  - [Dependency Installation](#dependency-installation-1)
  - [Training](#training)
- [Community and Ecosystem](#community-and-ecosystem)
- [Disclaimer, License and Citation](#disclaimer-license-and-citation)
  - [Disclaimer](#disclaimer)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Citation](#citation)

# Introduction

We are re-inventing the Large Language Model (LLM). This is the official implementation of TransNormerLLM in [link](https://arxiv.org/pdf/2307.14995.pdf. Our opened weights of TransNormerLLM are now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly.

Our release contains the TransNormerLLM model implementation, the open-source weights and the starting code for Supervised Fine-tuning (SFT). We will show examples on how to load [TransNormerLLM](https://github.com/OpenNLPLab/Transnormer) models, run SFT and inference on it.

- TransNormerLLM is the first linear attention-based LLM that outperforms conventional softmax attention-based models in terms of both accuracy and efficiency. It was trained on a high-quality corpus with up to **1.4 trillion** tokens.
- TransNormerLLM evolves from the previous linear attention architecture TransNormer by making advanced modifications that include LRPE positional embedding, Lightning Attention acceleration, new gating and normalization mechanisms.
- TransNormerLLM achieved competitive performance of its size on multiple well-approved Chinese, English, and multi-language general and domain-specific benchmarks.
- This release includes **Base** versions with **385M**, **1B**, and **7B** parameters.
- All versions are fully open to academic research. Developers only need to apply via email and obtain official commercial permission to use it for free commercially.
- For more information, welcome reading our academic paper [TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf).


# Released Weights

The specific released versions and download links are shown as below:

|         | Base Models  | 
|:-------:|:-----------:|
| 385M      | 🤗 [TransNormerLLM-385M](https://huggingface.co/OpenNLPLab/TransNormerLLM-385M) | 
| 1B     | 🤗 [TransNormerLLM-1B](https://huggingface.co/OpenNLPLab/TransNormerLLM-1B) |
| 7B    | 🤗 [TransNormerLLM-7B](https://huggingface.co/OpenNLPLab/TransNormerLLM-7B) | 

# Benchmark Results

To validate TransNormerLLM, we tested our 385M, 1B, and 7B models on Commonsense Reasoning Task, MMLU, CMMLU, and C-Eval. For comparison, we selected several open-source models as competitors, including Transformer-based models such as OPT, Pythia, BLOOM, GPT-Neo, GPT-J, MPT, Falcon, LLaMA1/2, OpenLLAMA v1/v2, Baichuan 1/2, ChatGLM 1/2, and non-Transformer model RWKV. It can be observed that, compared to these models, TransNormerLLM remains highly competitive.

**Commonsense Reasoning** We report BoolQ, PIQA, SIQA,
HellaSwag, WinoGrande, ARC easy and challenge, OpenBookQA and their average. We report 0-shot results for all benchmarks using LM-Eval-Harness.
All of our models achieve competitive performance compared to existing state-of-the-art LLMs, showcasing a remarkable ability to comprehend and apply commonsense reasoning.

**Aggregated Benchmarks**
We report the overall results for MMLU, CMMLU, C-Eval. Official scripts were used for evaluating MMLU, CMMLU, and C-Eval, with all evaluation results being conducted with a 5-shot setup. In comparison to top-tier open-source models available in the industry, our models have demonstrated matched performance in both English and Chinese benchmarks.

## General Domain

In the general domain, we conducted 5-shot tests on the following datasets:
- [C-Eval](https://cevalbenchmark.com/index.html#home) is a comprehensive Chinese basic model evaluation dataset, covering 52 disciplines and four levels of difficulty. Our evaluation approach followed that of [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness).
- [MMLU](https://arxiv.org/abs/2009.03300) is an English evaluation dataset comprising 57 tasks, encompassing elementary math, American history, computer science, law, etc. The difficulty ranges from high school level to expert level. It's a mainstream LLM evaluation dataset. We used its [official](https://github.com/hendrycks/test) evaluation approach.
- [CMMLU](https://github.com/haonan-li/CMMLU) is a comprehensive Chinese evaluation benchmark covering 67 topics, specifically designed to assess language models' knowledge and reasoning capabilities in a Chinese context. We adopted its [official](https://github.com/haonan-li/CMMLU) evaluation approach.


### Model Results
**Performance Comparison on Commonsense Reasoning and Aggregated Benchmarks.** For a fair comparison, we report competing methods' results reproduced by us using their released models. PS: parameter size (billion). T: tokens (trillion). HS: HellaSwag. WG: WinoGrande.

| Model       | PS   | T    | BoolQ          | PIQA           | HS             | WG             | ARC-e          | ARC-c          | OBQA           | MMLU           | CMMLU          | C-Eval         |
|-------------|------|------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| OPT         | 0.35 | 0.30 | 57.74          | 64.58          | 36.69          | 52.49          | 44.02          | 23.89          | 28.20          | 26.02          | 25.34          | 25.71          |
| Pythia      | 0.40 | 0.30 | 60.40          | 67.08          | 40.52          | 53.59          | 51.81          | 24.15          | 29.40          | 25.99          | 25.16          | 24.81          |
| BLOOM       | 0.56 | 0.35 | 55.14          | 64.09          | 36.97          | 52.80          | 47.35          | 23.98          | 28.20          | 24.80          | 25.35          | 27.14          |
| RWKV        | 0.43 | -    | -              | 67.52   | 40.90 | 51.14 | 52.86 | 25.17 | 32.40 | 24.85          | -              | -              |
| **Ours**        | 0.39 | 1.0  | 62.14          | 66.70          | 46.27          | 54.46          | 55.43          | 27.99          | 32.40          | 25.90          | 25.05          | 25.24          |


# Inference and Deployment

The model weights, source code, and configuration needed for inference have been released on Hugging Face. Download links can be found in the table at the beginning of this document. Below, we demonstrate various inference methods using TransNormerLLM-7B-Chat as an example. The program will automatically download the required resources from Hugging Face.

## Dependency Installation


**📝Note** Please configure the following environment before using the model:

```shell
pip install triton==2.0.0
pip install einops
```

## Notice
If you encounter errors related to Triton, please set the following environment variables:
```
export use_triton=False
```


## Python Code Inference

### Demonstration of Base Model Inference

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("OpenNLPLab/TransNormerLLM-385M", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("OpenNLPLab/TransNormerLLM-385M", device_map="auto", trust_remote_code=True)
```

> In the above code snippets, the model loading specifies `device_map='auto'`, which will use all available GPUs. If you need to specify the device(s) to use, you can control it in a way similar to `export CUDA_VISIBLE_DEVICES=0,1` (using the 0 and 1 graphics cards).


# Fine-tuning the Model

## Dependency Installation

```shell
git clone https://github.com/OpenNLPLab/TransNormerLLM.git
cd TransNormerLLM/fine-tune
pip install -r requirements.txt
```
- To use lightweight fine-tuning methods like LoRA, you must additionally install [peft](https://github.com/huggingface/peft).

## Training

Below, we provide an example of fine-tuning the TransNormerLLM-1B on a single machine with ZeRO-3.

Training Data: `alpaca_data.json`. This sample data was drawn from [alpaca_data.json](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json), consisting of a selection of 52,002 entries, and has been reformatted. The main purpose is to demonstrate how to SFT our model, and effectiveness is not guaranteed.

```shell
torchrun \
    --nproc_per_node=8 \
    train.py \
    --model_name_or_path OpenNLPLab/TransNormerLLM-385M \
    --data_path ./alpaca_data.json \
    --output_dir output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 true \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --deepspeed 'configs/zero3.json' \
    --logging_steps 1 \
    --dataloader_num_workers 24 \
    --ddp_find_unused_parameters false \
    --tf32 true \
```

# Community and Ecosystem

**📢📢📢 We will continuously update the support for TransNormerLLM from the community and ecosystem here 😀😀😀**
- [nanoTransnormer](https://github.com/Doraemonzzz/nanoTransNormer)

# Disclaimer, License and Citation

## Disclaimer
We hereby declare that our team has not developed any applications based on TransNormerLLM models, not on iOS, Android, the web, or any other platform. We strongly call on all users not to use TransNormerLLM models for any activities that harm national / social security or violate the law. Also, we ask users not to use TransNormerLLM models for Internet services that have not undergone appropriate security reviews and filings. We hope that all users can abide by this principle and ensure that the development of technology proceeds in a regulated and legal environment.

We have done our best to ensure the compliance of the data used in the model training process. However, despite our considerable efforts, there may still be some unforeseeable issues due to the complexity of the model and data. Therefore, if any problems arise due to the use of TransNormerLLM open-source models, including but not limited to data security issues, public opinion risks, or any risks and problems brought about by the model being misled, abused, spread or improperly exploited, we will not assume any responsibility.

## License
The community usage of TransNormerLLM model requires adherence to [Apache 2.0](https://github.com/OpenNLPLab/TransNormerLLM/blob/main/LICENSE) and [Community License for TransNormerLLM Model](https://huggingface.co/OpenNLPLab/TransNormerLLM-385M/blob/main/TransNormerLLM模型社区许可协议.pdf). The TransNormerLLM model supports commercial use. If you plan to use the TransNormerLLM model or its derivatives for commercial purposes, please ensure that your entity meets the following conditions:

  1. The Daily Active Users (DAU) of your or your affiliate's service or product is less than 1 million.
  2. Neither you nor your affiliates are software service providers or cloud service providers.
  3. There is no possibility for you or your affiliates to grant the commercial license given to you, to reauthorize it to other third parties without TransNormerLLM's permission.

Upon meeting the above conditions, you need to submit the application materials required by the TransNormerLLM Model Community License Agreement via the following contact email: opennlplab@gmail.com. Once approved, TransNormerLLM will hereby grant you a non-exclusive, global, non-transferable, non-sublicensable, revocable commercial copyright license.

## Acknowledgments
Our project is developed based on the following open source projects:
- [Baichuan](https://github.com/baichuan-inc/Baichuan-7B) for the tokenizer.
- [metaseq](https://github.com/facebookresearch/metaseq) for training.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

## Citation
If you wish to cite our work, please use the following reference:
```
@article{qin2023scaling,
  title={Scaling transnormer to 175 billion parameters},
  author={Qin, Zhen and Li, Dong and Sun, Weigao and Sun, Weixuan and Shen, Xuyang and Han, Xiaodong and Wei, Yunshen and Lv, Baohong and Yuan, Fei and Luo, Xiao and others},
  journal={arXiv preprint arXiv:2307.14995},
  year={2023}
}
```
