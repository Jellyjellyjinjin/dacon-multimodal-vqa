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
## 2-1. input_train(csv to json)

 LLaVA/llava/train/train.py 

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
## 2-2. input_test(json to jsonl)

 LLaVA/llava/eval/model_vqa.py

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
  llava/train/train.py
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
  * LLaVA/llava/model/builder.py     
     !![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/1ed6a2db-6d7b-4c02-a95c-00b3d920c55b)

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
-----------------------------------------------------------------------------------------------------------
## Visual Instruction tuning Zero_shot

```python
!python3 /content/LLaVA/llava/eval/model_vqa.py \
    --model-path liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3\
    --question-file \
    /content/test13.jsonl \
    --image-folder \
   /content/image/test \
    --answers-file \
    /content/result14.jsonl
```
**output**
![image](https://github.com/Jellyjellyjinjin/dacon-multimodal-vqa/assets/118363210/3e7ddcdf-473d-4d6c-9026-f27772752dcd)


**1. Devide question**
1.   What color?  ㅡ>  def color () : 문장안에서 색에 해당하는 단어만 추출 
2.   How many? ㅡ> def num() : 문장안에서 숫자에 해당하는 단어만 추출 
3. etc.

```python
def process_sentence(question, answer):
    if question.startswith("What color"):
        # "What color"로 시작하는 경우
        return color(answer, rainbow_colors)
    elif question.startswith("How many"):
        # "How many"로 시작하는 경우
        return num(answer, numword)
    else:
        # 위 두 경우에 해당하지 않는 경우
        return answer
```
**2. Remove answer's stopword**
```python
def remove_articles(text):

    # 소문자로 변환
    text = text.lower()

    # 정규 표현식을 사용하여 관사 제거
    text = re.sub(r'\b(a|an|the|in|on|at|by|with|from|)\b', '', text)

    # 여러 개의 공백을 단일 공백으로 대체
    text = re.sub(r'\s+', ' ', text).strip()

    # 선택된 단어를 다시 결합
    result_text = ''.join(text)

    return result_text
```
**3.Remove duplicate words in the answer**

```python
from tqdm import tqdm

def remove_words_from_b_if_in_a(sentence_a, sentence_b):
    # 문장 A와 B를 공백을 기준으로 단어로 분리
    sentencea_a = sentence_a[:-1]
    words_a = sentencea_a.split()
    words_b = sentence_b.split()

    if all(word in 'or' and word in 'and' and word in 'many' for word in words_a):
      df2 = [i for i  in words_b if i not in words_a]
      return df2
    else:
      return sentence_b


processed_answers = [remove_articles(answer) for answer in  processed_answers]
```
**4.Extract answer's keywords**
```python
from tqdm import tqdm

plzan = []

for qdoc, adoc in tqdm(zip(json_data2, processed_answers)):
    if " and " in adoc:
      if len(adoc.split()) >= 5:
        adoc = remove_words_from_b_if_in_a(qdoc,adoc)
        a = kw_model.extract_keywords(adoc, keyphrase_ngram_range=(1,1), stop_words=None)
        plzan.append(a[0][0])
      else :
        plzan.append(adoc)
    else :
      adoc = remove_words_from_b_if_in_a(qdoc,adoc)
      a = kw_model.extract_keywords(adoc, keyphrase_ngram_range=(1,1), stop_words=None)
      plzan.append(a[0][0])
```
**5.convert string to num**
```python
def convert_word_to_number(text):
    words = text.lower().split()
    for word in words:
        if word in number_words_dict:
            return str(number_words_dict[word])
    return text
```
