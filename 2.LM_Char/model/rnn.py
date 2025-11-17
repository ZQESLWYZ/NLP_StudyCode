import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter
import torch.optim as optim

class ChineseRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 n_layers=2, dropout=0.3):
        super(ChineseRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 嵌入层：将字符索引转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN层
        self.lstm = nn.GRU(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # 嵌入层
        x = self.embedding(x)
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出
        lstm_out = self.dropout(lstm_out[:, -1, :])
        
        # 全连接层
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_dim))

class TextGenerator:
    def __init__(self, model, char2idx, idx2char):
        self.model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.model.eval()
    
    def generate_text(self, start_text, length=100, temperature=1.0, device='cpu'):
        """生成文本
        Args:
            start_text: 生成的起始文本
            length: 要生成的文本长度
            temperature: 采样温度
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.model.eval()
        
        # 初始化输入
        generated = list(start_text)
        input_seq = torch.tensor([[self.char2idx[char] for char in start_text[-50:]]], 
                                dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(length):
                # 前向传播
                output, _ = self.model(input_seq)
                
                # 应用温度参数
                output = output / temperature
                probabilities = torch.softmax(output, dim=1)
                
                # 采样下一个字符
                predicted = torch.multinomial(probabilities, 1).item()
                next_char = self.idx2char[predicted]
                
                generated.append(next_char)
                
                # 更新输入序列
                input_seq = torch.cat([input_seq[:, 1:], 
                                     torch.tensor([[predicted]], dtype=torch.long, device=device)], dim=1)
        
        return ''.join(generated)