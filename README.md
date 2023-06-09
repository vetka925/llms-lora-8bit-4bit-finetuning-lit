# LLMs LORA Finetuning

Research a finetuning of LLMs with LORA.

* LLaMA 7B and 13B, 8 and 4 bit finetuned with LORA and pytorch lit. 
* GPT-J 7B 8 bit finetuning.
* Saving and loading finutened models.
* Evaluating few-shot method.
* Using finetuned models in new task.
* Comparing models perfomance with ChatGPT and other OpenAI models


The purpose of this repo to make little research of GPT like models and approaches to finetune quantized LLMs. Сlassification task was chosen as a test task. I compared accuracy of different setups. Also, I compared final metrics with metrics of OpenAI GPT models. This code can be reused to finutene GPT like models.  

[WanDB Report](https://api.wandb.ai/links/vetka925/qaqx6fhy)  

## System requirements

1. At least 11 GB of VRAM
2. Linux (required for bitsandbytes package)

This code was tested on WSL Ubuntu 22.04, Geforce GTX 1080 TI, Cuda toolkit 11.7

## Usage

To reproduce results locally:

1. Prepare environment
```bash
sudo apt update && sudo apt install git build-essential -y
conda install cuda=11.7.1 -c nvidia -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c conda-forge cudatoolkit=11.7 ninja accelerate sentencepiece -y
```
2. Clone repo
```bash
git clone https://github.com/vetka925/llms-lora-8bit-4bit-finetuning-lit
```
3. Install requirements**
```bash
cd gpt-j-8bit-lightning-finetune
pip install -r requirements.txt
```
4. Install CUDA extension for 4bit operations
```bash
python setup_cuda.py install
```
5. Run Jupyter notebook finetune.ipynb
```bash
jupyter notebook
```
**For possible issues with bitsandbytes on WSL use [this](https://github.com/TimDettmers/bitsandbytes/issues/112#issuecomment-1406329180)  

## Description

Full research description on [Medium](), [Habr](https://habr.com/ru/companies/neoflex/articles/722584/)

Finetuning: [finetune.ipynb](https://github.com/vetka925/llms-lora-8bit-4bit-finetuning-lit/blob/master/finetune.ipynb)  
Finetuning OpenAI model: [compare_openai.ipynb](https://github.com/vetka925/llms-lora-8bit-4bit-finetuning-lit/blob/master/compare_openai.ipynb)  
Load finetuned models and validate new data: [inference_finetuned.ipynb](https://github.com/vetka925/llms-lora-8bit-4bit-finetuning-lit/blob/master/inference_finetuned.ipynb)  
Fewshot example: [fewshot.ipynb](https://github.com/vetka925/llms-lora-8bit-4bit-finetuning-lit/blob/master/fewshot.ipynb)
  
Test task is **Hate Speech and Offensive Language Detection**.  
Data: 1000 train and 200 validation samples with balanced classes from [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)