import os
from transformers import AutoConfig, AutoModel, AutoTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897' 
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

model_name = "hfl/rbt3"

# 1.下载模型并展示模型参数
model = AutoModel.from_pretrained(model_name)
print(model.config) # 这是模型结构参数

# 2.更加详细的Config
config = AutoConfig.from_pretrained(model_name)
print(config.output_hidden_states) # 就是这个方法后有很多属性

# 3.不带头的模型调用
sen = '小小的我也有大大的梦想'

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(sen, return_tensors='pt')
# print(model(**inputs))

model = AutoModel.from_pretrained(model_name, output_attentions=True, hidden_states=True)

print(model(**inputs))

output = model(**inputs)

print(output.last_hidden_state.shape)

print(output.pooler_output.shape) # [CLS] 标记的位置的池化输出

print(output.hidden_states.shape)