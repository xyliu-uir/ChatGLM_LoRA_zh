# ChatGLM_LoRA_zh
在ChatGLM大模型上利用LoRA方法进行小参数学习，训练语料库选择中文的[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)

## 确认环境

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
