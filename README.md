# 월간 데이콘 이미지 기반 질의 응답 AI 경진대회
## 결과
* **★ PUBLIC 3위 / PRIVATE 2위 ★** 
* LLaVA 모델 from pretrained train data

## 1. Introduction
**[배경]**
* 멀티모달 AI는 서로 다른 유형의 데이터를 결합하여 사용하는 기술로, 텍스트와 이미지 등 다양한 데이터를 종합적으로 다루는 기술입니다.
* 서비스적으로 활용 가치가 높은 멀티모달 AI 모델 개발 및 고도화에 도전해 보세요!

**[주제]** 이미지 기반 질의 응답 AI 모델 개발

**[기간]** 2023.07.10. ~ 2023.08.07.

**[링크]** https://dacon.io/competitions/official/236118/overview/description

## 2. Data
```
data
├─  image
│   ├─  train : 107,231개
│   │   ├─  train_000000.png
│   │   ├─  train_000001.png
│   │   └─  ...
│   └─  test : 11,915개
│       ├─  test_00000.png
│       ├─  test_00001.png
│       └─  ...
├─  train.csv
|    ├─  ID : 질문 ID
|    ├─  image_id : 이미지 ID
|    ├─  question : 이미지 관련 질문
|    └─  answer : 질문에 대한 답변
├─  test.csv
|    ├─  ID : 질문 ID
|    ├─  image_id : 이미지 ID
|    └─  question : 이미지 관련 질문
└─  sample_submission.csv
     ├─  ID : 질문 ID
     └─  *answer : 질문에 대한 답변
```
## 2-1. input_train

**CHECK**  LLaVA/llava/train/train.py 

![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/d526ad41-bac4-45a1-84e3-2deb8b8ac33e)

```python
import zipfile
import os
import csv
import json

# ----------------------------------------------------------------
# make 'output.json'
with open('/content/dacon-multimodal-vqa/train.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

json_data = []
for row in data:
    id, image_id, question, answer = row
    json_data.append({
        "id": id,
        "image": "/content/dacon-multimodal-vqa/image/train/" + image_id + ".jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + question
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    })

with open('output.json', 'w') as f:
    json.dump(json_data, f, indent=4)
```
![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/7d3aa577-6701-4340-8274-b12e1da43f80)
## 2-2. input_test

**CHECK** LLaVA/llava/eval/model_vqa.py

![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/41604dd4-c0a8-453e-980c-2528fe467059)

```python
# make 'test.json'
with open('/content/test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

json_data = []
for row in data:
    id, image_id, question = row
    json_data.append({
        "question_id": id,
        "image": "/content/image/test/" + image_id + ".jpg",
        "text": question
        })

# jsonl file path
jsonl_output_file = "/content/test.jsonl"

# JSON to JSONL 
with open(jsonl_output_file, "w") as file:
    for obj in json_data:
        # write file (JSON +(\n)).
        json.dump(obj, file)
        file.write("\n")

```
## 2-3. output_test
**EDIT** LLaVA/llava/eval/model_vqa.py
 ![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/ad900fec-fe9a-4c57-83cc-ddd3f5695749)

## 3. Setup
* In Colab-PRO or PRO+ Users only
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


## 4. Run
* For recording wandb
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
* output_dir folder should be contained **'checkpoint-*'**
* num_train_epochs must have started from **2** or more
 **CHECK** llava/train/train.py
   ![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/a6705c39-567a-43ec-ba9b-92bd4d793cd2)



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
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
  
## 6. Inference
**EDIT** LLaVA/llava/model/builder.py
![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/08f9f546-16a4-4866-8c22-55b9eda57ece)


* You should change output_dir name 'checkpoint-*' to **'LLaVA-version'**
  * May be you might get a difference whether the name contains 'LLaVA' or not
  * **CHECK**LLaVA/llava/model/builder.py     
     !![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/1ed6a2db-6d7b-4c02-a95c-00b3d920c55b)


```python
%cd /content

# go to your output directory
from google.colab import drive
drive.mount('/content/drive')
```
```python
%cd /content/LLaVA
!python /content/dacon-multimodal-vqa/eval/model_vqa.py \
    --model-path /content/drive/MyDrive/llava/checkpoint/LLaVA-7B-v1.3 \
    --model-base lmsys/vicuna-7b-v1.3 \
    --question-file \
    /content/dacon-multimodal-vqa/test.jsonl \
    --image-folder \
   /content/image/test \
    --answers-file \
    /content/result.jsonl \
```

## 7. Submission
```python
%cd /content/dacon-multimodal-vqa
!python submission.py
```
