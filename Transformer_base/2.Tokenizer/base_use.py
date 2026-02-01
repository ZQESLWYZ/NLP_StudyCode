import os
from transformers import AutoTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897' 
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

sen = "小小的我也有大大的梦想！"

# 1.分词器加载与本地保存

model_name = "uer/roberta-base-finetuned-dianping-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained("./roberta_tokenizer")

tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer") # 从本地加载分词器

# 2.句子分词

tokens = tokenizer.tokenize(sen)
print(tokens)

# 查看词典索引映射关系
# print(tokenizer.vocab)
print(tokenizer.vocab_size)

# 3.token_ids互相转换

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)

# 更简便的编码解码实现方法
ids = tokenizer.encode(sen, add_special_tokens=True) # 句子开头结尾会增加[CLS]和[SEP]开始结束标志
print(ids)

tokens = tokenizer.decode(ids)
print(tokens)

# 4.填充与截断，仍然基于encode和decode

# 不够max_length -> 填充
ids = tokenizer.encode(sen, padding="max_length", max_length=20) # 必须加max_length参数不然不会填充
print(ids)

# 超过max_length -> 阶段
ids = tokenizer.encode(sen, max_length=5)
print(ids)

# 5.其它输入部分
# attention_mask（用于屏蔽填充idx） 和 type_id（区分不同句子，多句情况）

inputs = tokenizer._encode_plus(sen, padding="max_length", max_length=10)
print(inputs)

inputs = tokenizer(sen, padding="max_length", max_length=10)
print(inputs)

# 6.处理Batch数据

sens = ["小小的我也有大大的梦想！",
        "河南科技学院计算机科学与技术学院",
        "HISTWheatSeed"]

res = tokenizer(sens, padding="max_length", max_length=15)
print(res)

# 7.特殊原创Tokenizer的加载
# 有些模型的分词器是自己实现的，要加载需要trust
s_tokenizer = AutoTokenizer.from_pretrained("zai-org/chatglm3-6b", trust_remote_code=True)

s_tokenizer.save_pretrained("./chatglm6b")

print(s_tokenizer(sens))

