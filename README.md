# ChatGLM_LoRA_zh
在ChatGLM大模型上利用LoRA方法进行小参数学习，训练语料库选择中文的[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)

本仓库是使用LoRA复现清华大学+智谱AI的ChatLM语言模型结果的代码。大部分参考[ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)，感谢大佬

Paper:

[LoRA](https://arxiv.org/pdf/2106.09685.pdf)

[GLM](http://arxiv.org/abs/2103.10360)

[GLM-130B](http://arxiv.org/abs/2210.02414)
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

另：CUDA版本11.3（建议>11.6），python版本3.8，显卡3090-24G

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

## 四、下载GLM模型（视网络情况选择）
若网络情况良好，请直接使用：
```python
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map="auto")
```
会自动开始下载模型，下载完的模型会放在~/.cache/huggingface中

若网络情况不佳，可以手动下载模型，并修改model和tokenizer的路径
从[THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)上下载所有的模型文件放到glm_models文件夹中，约13.5GB，请预留足够的磁盘空间。
```python
tokenizer = AutoTokenizer.from_pretrained("./glm_models", trust_remote_code=True)
model = AutoModel.from_pretrained("./glm_models", load_in_8bit=True, trust_remote_code=True, device_map="auto")
```

## 五、开始Fine-tune
```bash
python finetune.py --dataset_path data/alpaca-zh --lora_rank 8 --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --max_steps 20380 --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 50 --output_dir output
```
![image](https://user-images.githubusercontent.com/50279789/230530922-b9f7c882-f752-4e4d-9bee-b927b17219b3.png)

```
大概训练需要10个小时左右，训练完成后，会在output文件夹下生成output/adapter_model.bin文件，如果想跳过训练步骤，可以直接使用这个文件进行推理。
```

## 六、推理
请执行[infer.ipynb](infer.ipynb)

推理结果

![image](https://user-images.githubusercontent.com/50279789/230532652-73474857-27db-436e-a5b6-e0a3aef3d70b.png)

