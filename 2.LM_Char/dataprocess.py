import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter
import torch.optim as optim

class ChineseTextProcessor:
    def __init__(self, file_path, seq_length=50):
        self.file_path = file_path
        self.seq_length = seq_length
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        
    def load_and_clean_text(self):
        """加载并清洗中文文本"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 清洗文本：保留中文字符、标点符号和基本符号
        text = re.sub(r'[^\u4e00-\u9fa5，。！？；：""（）《》\s]', '', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def build_vocabulary(self, text):
        """构建字符词典"""
        # 统计字符频率
        char_counter = Counter(text)
        # 创建字符到索引的映射
        chars = sorted(char_counter.keys())
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(chars)
        
        print(f"词汇表大小: {self.vocab_size}")
        return self.char2idx, self.idx2char
    
    def text_to_sequences(self, text):
        """将文本转换为数字序列"""
        sequences = []
        next_chars = []
        
        for i in range(0, len(text) - self.seq_length, 1):
            seq = text[i:i + self.seq_length]
            next_char = text[i + self.seq_length]
            
            sequences.append([self.char2idx[char] for char in seq])
            next_chars.append(self.char2idx[next_char])
        
        return np.array(sequences), np.array(next_chars)