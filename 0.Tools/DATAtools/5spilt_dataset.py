import json
import random
from typing import List, Dict, Any
import torch

# ============ 1. 加载 SentencePiece 模型 ============
import sentencepiece as spm

class SPTokenizer:
    """封装 SentencePiece 模型，提供 encode/decode 接口"""
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()
        # 特殊 token IDs（根据你的训练设置，可能需要手动指定）
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 0
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() != -1 else 1
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 2
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 3

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = self.sp.encode_as_ids(text)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = False) -> str:
        if remove_special_tokens:
            ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode_ids(ids)


# 加载你的模型（改成实际路径）
src_tokenizer = SPTokenizer(r"0.Tools\DATAtools\result\spm_en.model")   # 英文模型
tgt_tokenizer = SPTokenizer(r"0.Tools\DATAtools\result\spm_zh.model")   # 中文模型


# ============ 2. 读取原始平行语料（带诊断） ============
def read_parallel_corpus_debug(src_path: str, tgt_path: str):
    # 读取原始行（保留换行符）
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = [line for line in f]
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = [line for line in f]

    # 去掉换行符 \n 和 \r，得到纯文本行
    src_stripped = [line.rstrip('\n\r') for line in src_lines]
    tgt_stripped = [line.rstrip('\n\r') for line in tgt_lines]

    # 过滤掉 strip() 后为空的行（即空行或只有空白字符的行）
    src_non_empty = [line for line in src_stripped if line.strip() != '']
    tgt_non_empty = [line for line in tgt_stripped if line.strip() != '']

    # 打印诊断信息
    print(f"[诊断] 原始行数: src={len(src_lines)}, tgt={len(tgt_lines)}")
    print(f"[诊断] 去掉换行符后行数: src={len(src_stripped)}, tgt={len(tgt_stripped)}")
    print(f"[诊断] 非空行数: src={len(src_non_empty)}, tgt={len(tgt_non_empty)}")

    # 找出第一个内容不同的地方
    min_len = min(len(src_non_empty), len(tgt_non_empty))
    found_diff = False
    for i in range(min_len):
        if src_non_empty[i] != tgt_non_empty[i]:
            print(f"[诊断] 第 {i+1} 行内容不同:")
            print(f"  src: {repr(src_non_empty[i][:100])}...")
            print(f"  tgt: {repr(tgt_non_empty[i][:100])}...")
            found_diff = True
            break

    if not found_diff and len(src_non_empty) != len(tgt_non_empty):
        print(f"[诊断] 前 {min_len} 行内容一致，但行数不同:")
        if len(src_non_empty) > len(tgt_non_empty):
            print(f"  src 多出的一行 (#{min_len+1}): {repr(src_non_empty[min_len][:100])}")
        else:
            print(f"  tgt 多出的一行 (#{min_len+1}): {repr(tgt_non_empty[min_len][:100])}")

    # 安全配对：取最小长度
    pairs = list(zip(src_non_empty[:min_len], tgt_non_empty[:min_len]))
    print(f"[诊断] 成功配对 {len(pairs)} 条句对。")
    return pairs


# ============ 3. 分词编码 ============
def encode_corpus(pairs, src_tokenizer, tgt_tokenizer, max_len=512):
    data = []
    for src_sent, tgt_sent in pairs:
        src_ids = src_tokenizer.encode(src_sent, add_bos=False, add_eos=True)
        tgt_ids = tgt_tokenizer.encode(tgt_sent, add_bos=False, add_eos=True)
        # 截断
        if len(src_ids) > max_len:
            src_ids = src_ids[:max_len]
        if len(tgt_ids) > max_len:
            tgt_ids = tgt_ids[:max_len]
        data.append({
            "src_text": src_sent,
            "tgt_text": tgt_sent,
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        })
    return data


# ============ 4. 划分训练/验证/测试集 ============
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    划分训练集、验证集、测试集
    
    Args:
        data: 编码后的数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例  
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_data, val_data, test_data: 三个数据集
    """
    # 检查比例总和是否为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"训练集、验证集、测试集比例之和应为1，当前为{total_ratio}")
    
    random.seed(seed)
    random.shuffle(data)
    
    total_size = len(data)
    train_end = int(total_size * train_ratio)
    val_end = int(total_size * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


# ============ 5. 保存数据集 ============
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_pt(data, path):
    src_ids_list = [item['src_ids'] for item in data]
    tgt_ids_list = [item['tgt_ids'] for item in data]
    src_texts = [item['src_text'] for item in data]
    tgt_texts = [item['tgt_text'] for item in data]
    torch.save({
        'src_ids': src_ids_list,
        'tgt_ids': tgt_ids_list,
        'src_text': src_texts,
        'tgt_text': tgt_texts
    }, path)


# ============ 主流程 ============
if __name__ == "__main__":
    # 文件路径（根据你的实际情况修改）
    src_file = r"5.Transformer\data\corpus.en"   # 英文原始句子
    tgt_file = r"5.Transformer\data\corpus.ch"   # 中文原始句子

    # 1. 读取平行语料（带诊断）
    print("读取平行语料...")
    pairs = read_parallel_corpus_debug(src_file, tgt_file)
    print(f"共 {len(pairs)} 条句对")

    # 2. 分词编码
    print("分词编码...")
    encoded_data = encode_corpus(pairs, src_tokenizer, tgt_tokenizer, max_len=256)

    # 3. 划分训练/验证/测试集
    print("划分数据集...")
    # 可以调整比例，例如：train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    train_data, val_data, test_data = split_data(
        encoded_data, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1,
        seed=42
    )

    # 4. 保存
    print("保存数据集...")
    # 确保目录存在
    import os
    os.makedirs("dataset", exist_ok=True)

    # 保存JSON格式
    save_json(train_data, "dataset/train_dataset.json")
    save_json(val_data, "dataset/val_dataset.json")      # 新增验证集
    save_json(test_data, "dataset/test_dataset.json")
    
    # 保存PyTorch格式
    save_pt(train_data, "dataset/train_dataset.pt")
    save_pt(val_data, "dataset/val_dataset.pt")          # 新增验证集
    save_pt(test_data, "dataset/test_dataset.pt")

    # 打印统计信息
    print(f"训练集: {len(train_data)} 条 ({len(train_data)/len(encoded_data)*100:.1f}%)")
    print(f"验证集: {len(val_data)} 条 ({len(val_data)/len(encoded_data)*100:.1f}%)")
    print(f"测试集: {len(test_data)} 条 ({len(test_data)/len(encoded_data)*100:.1f}%)")
    print("完成！")