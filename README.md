# 월간 데이콘 이미지 기반 질의 응답 AI 경진대회

## 1. Introduction

## 2. Data

## 3. Setup
* In Colab-PRO or + Users only
* Set up for sure GPU A100

### Clone LLaVA
```python
!git clone https://github.com/haotian-liu/LLaVA.git
%cd /content/LLaVA
```

### Install
```python
!pip install --upgrade pip
!pip install -e .
!pip install ninja
!pip install flash-attn --no-build-isolation
```

### Clone Vicuna
```python
!git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
```

### Download Data
```python
# Download directly
!gdown https://drive.google.com/u/0/uc?id=1a9XB3r83ZCFWLOHBp8ooz3zQFl9rEIei&export=download
```

### Preprocessing
* You could get 'output.json' file
* If else, get down our file and run it in your '/content' directory
```python
%cd /content
!git clone https://github.com/pimang62/dacon-multimodal-vqa.git

%cd /content/dacon-multimodal-vqa
!python preprocessing.py
```

## 4. Run
* Record wandb
  * put your API
```python
%cd /content/LLaVA
!pip install wandb
!wandb login
```

* Train
```python
!python /content/LLaVA/llava/train/train_mem.py \
    --model_name_or_path /content/LLaVA/vicuna-7b-v1.3 \
    --version v1 \
    --data_path /content/dacon-multimodal-vqa/output.json \
    --image_folder /content/dacon-multimodal-vqa/image/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir /content/drive/MyDrive/llava \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

## 5. Re-training
* You should put 'vicuna' to your model-name
* output_dir name should be contained 'checkpoint-*'
* num_train_epochs must have started from 2

```python
!python /content/LLaVA/llava/train/train_mem.py \
    --model_name_or_path /content/LLaVA/vicuna-7b-v1.3\
    --version v1 \
    --data_path /content/dacon-multimodal-vqa/output.json \
    --image_folder /content/dacon-multimodal-vqa/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir /content/drive/MyDrive/llava/checkpoint-2400 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.00 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

## 6. Inference

```python
%cd /content

# go to your output directory
from google.colab import drive
drive.mount('/content/drive')
```

* You should change output_dir 'checkpoint-*' to 'llava-*"
  * May be you might get a difference whether the name contains 'llava' or not

```python
%cd /content/LLaVA
!python3 /content/dacon-multimodal-vqa/eval/model_vqa.py \
    --model-path /content/drive/MyDrive/llava/checkpoint/llava-2400 \
    --model-base lmsys/vicuna-7b-v1.3 \
    --question-file \
    /content/dacon-multimodal-vqa/test.jsonl \
    --image-folder \
   /content/image/test \
    --answers-file \
    /content/result.jsonl \
```

## Submission
```python
%cd /content
!python submission.py
```
