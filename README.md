# ChatGLM_LoRA_zh
在ChatGLM大模型上利用LoRA方法进行小参数学习，训练语料库选择中文的[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)

## 一、确认环境

bitsandbytes==0.37.0

accelerate==0.17.1

protobuf>=3.19.5,<3.20.1

transformers==4.27.1

icetk

cpm_kernels==1.0.11

torch>=1.13.1

tensorboard

datasets==2.10.1

git+https://github.com/huggingface/peft.git  # 最新版本 >=0.3.0.dev0

或者直接进入工程目录，pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

另：CUDA版本11.3，python版本3.8，显卡3090-24G

## 二、将项目克隆到本地
## 三、数据预处理
### 3.1 将data/alpaca-zh.json转化为jsonl格式
 ```bash
 python cover_alpaca2jsonl.py --data_path data/alpaca-zh.json --save_path data/alpaca-zh.jsonl
 ```
 
 ### 3.2分词
 ```bash
 python tokenize_dataset_rows.py --jsonl_path data/alpaca-zh.jsonl --save_path data/alpaca-zh --max_seq_length 200
 ```
其中 
- `--jsonl_path` 微调的数据路径, 格式jsonl, 对每行的['context']和['target']字段进行encode
- `--save_path` 输出路径
- `--max_seq_length` 样本的最大长度

## 四、开始Fine-tune
```bash
python finetune.py --dataset_path data/alpaca --lora_rank 8 --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --max_steps 23800 --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 50 --output_dir output
```
