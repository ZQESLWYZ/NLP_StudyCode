import os
import json
from tqdm import tqdm
from sentencepiece import SentencePieceTrainer
import sentencepiece as spm

# ================== 配置参数 ==================

# 输入文件路径（平行语料，建议已经做过基本清洗）
EN_INPUT_FILE = r'5.Transformer\data\corpus.en'     # 英文文本，一行一句
ZH_INPUT_FILE = r'5.Transformer\data\corpus.ch'     # 中文文本，一行一句

# 输出目录，用于保存 tokenizer 模型和分词结果
OUTPUT_DIR = 'tokenizer_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 词表大小设定
EN_VOCAB_SIZE = 30000
ZH_VOCAB_SIZE = 32000

# SentencePiece 模型前缀（输出文件名前缀）
EN_SPM_MODEL_PREFIX = os.path.join(OUTPUT_DIR, 'spm_en')  # 英文模型
ZH_SPM_MODEL_PREFIX = os.path.join(OUTPUT_DIR, 'spm_zh')  # 中文模型

# 编码后的输出文件
EN_TOKENIZED_FILE = os.path.join(OUTPUT_DIR, 'en_tokenized.txt')
ZH_TOKENIZED_FILE = os.path.join(OUTPUT_DIR, 'zh_tokenized.txt')

# ===============================================
# Step 1: 英文分词 - 使用 SentencePiece（Unigram 算法）
# ===============================================

print("Step 1: 训练英文 SentencePiece Tokenizer（Unigram）")

SentencePieceTrainer.train(
    input=EN_INPUT_FILE,
    model_prefix=EN_SPM_MODEL_PREFIX,
    vocab_size=EN_VOCAB_SIZE,
    model_type='unigram',                    # 使用 Unigram 算法
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
    user_defined_symbols=['[CLS]', '[SEP]', '[MASK]'],  # 添加特殊 token
    train_extremely_large_corpus=True,
    character_coverage=1.0,                  # 英文建议 1.0（覆盖所有 ASCII/Unicode）
    input_sentence_size=0,                   # 0 表示使用全部语料
    shuffle_input_sentence=True
)

print(f"英文 tokenizer 模型已保存为：{EN_SPM_MODEL_PREFIX}.model 和 .vocab")

# 加载英文模型并进行编码
sp_en = spm.SentencePieceProcessor()
sp_en.load(f"{EN_SPM_MODEL_PREFIX}.model")

with open(EN_INPUT_FILE, 'r', encoding='utf-8') as fin, \
     open(EN_TOKENIZED_FILE, 'w', encoding='utf-8') as fout:
    for line in tqdm(fin, desc="英文编码"):
        line = line.strip()
        if not line:
            continue
        pieces = sp_en.encode_as_pieces(line)  # 如 ['▁I', '▁love', '▁NLP', '.']
        fout.write(' '.join(pieces) + '\n')

print(f"英文编码完成，保存至：{EN_TOKENIZED_FILE}")

# ===============================================
# Step 2: 中文分词 - 使用 SentencePiece（Unigram 算法）
# ===============================================

print("Step 2: 训练中文 SentencePiece Tokenizer（Unigram）")

SentencePieceTrainer.train(
    input=ZH_INPUT_FILE,
    model_prefix=ZH_SPM_MODEL_PREFIX,
    vocab_size=ZH_VOCAB_SIZE,
    model_type='unigram',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
    user_defined_symbols=['[CLS]', '[SEP]', '[MASK]'],
    train_extremely_large_corpus=True,
    character_coverage=0.9995,               # 中文建议 0.9995（覆盖 CJK 扩展区）
    input_sentence_size=0,
    shuffle_input_sentence=True
)

print(f"中文 tokenizer 模型已保存为：{ZH_SPM_MODEL_PREFIX}.model 和 .vocab")

# 加载中文模型并进行编码
sp_zh = spm.SentencePieceProcessor()
sp_zh.load(f"{ZH_SPM_MODEL_PREFIX}.model")

with open(ZH_INPUT_FILE, 'r', encoding='utf-8') as fin, \
     open(ZH_TOKENIZED_FILE, 'w', encoding='utf-8') as fout:
    for line in tqdm(fin, desc="中文编码"):
        line = line.strip()
        if not line:
            continue
        pieces = sp_zh.encode_as_pieces(line)  # 如 ['▁自然', '▁语言', '▁处理']
        fout.write(' '.join(pieces) + '\n')

print(f"中文编码完成，保存至：{ZH_TOKENIZED_FILE}")

# ===============================================
# ✅ 结束
# ===============================================

print("\n✅ 分词完成（中英文均使用 Unigram）：")
print(f"- 英文 Unigram tokenizer 保存在: {EN_SPM_MODEL_PREFIX}.model")
print(f"- 中文 Unigram tokenizer 保存在: {ZH_SPM_MODEL_PREFIX}.model")
print(f"- 分词结果：")
print(f"  英文 -> {EN_TOKENIZED_FILE}")
print(f"  中文 -> {ZH_TOKENIZED_FILE}")