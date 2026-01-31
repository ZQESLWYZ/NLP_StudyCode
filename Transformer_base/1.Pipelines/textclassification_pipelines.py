import gradio as gr
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897' 
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

from transformers.pipelines import SUPPORTED_TASKS

# 查看支持的任务类型
# print(SUPPORTED_TASKS.items())
for k, v in SUPPORTED_TASKS.items():
    print(f"{k}")
    
# 如何创建一个pipeline?
from transformers import *

# 第一种方式：指定任务+模型
pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese", device=0)

print(pipe(r"商品不好用"))

# 第二种方式：预先加载模型，再创建Pipeline
model_name = "uer/roberta-base-finetuned-dianping-chinese"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe1 = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
print(pipe1(r"商品不好用", top_k=10))

print(pipe1.model.device)

# 如何确定pipeline的参数?
print(pipe1) # TextClassificationPipeline
print(TextClassificationPipeline)





