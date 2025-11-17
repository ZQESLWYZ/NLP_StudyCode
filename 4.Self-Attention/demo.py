import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # 存储注意力权重用于可视化
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数 (QK^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # 存储用于可视化
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到V
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出线性变换
        output = self.w_o(context)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SingleLayerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, dropout=0.1):
        super(SingleLayerTransformer, self).__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 多头自注意力层
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 层归一化和前馈网络（简化版）
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 自注意力
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        return x
    
    def get_attention_weights(self):
        """获取注意力权重用于可视化"""
        if self.self_attention.attention_weights is not None:
            return self.self_attention.attention_weights
        return None

def create_padding_mask(seq, pad_token_id=0):
    """创建padding掩码"""
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(size):
    """创建因果掩码（用于语言模型）"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def plot_attention_weights(attention_weights, sentence_tokens, head_idx=0, layer_name="Self-Attention"):
    """绘制单个注意力头的权重热力图"""
    # 取第一个batch，指定注意力头
    attn_weights = attention_weights[0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, 
                xticklabels=sentence_tokens,
                yticklabels=sentence_tokens,
                cmap="YlGnBu",
                annot=True,  # 显示数值
                fmt=".2f",
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f"{layer_name} - Head {head_idx + 1}")
    plt.xlabel("Key (Source Tokens)")
    plt.ylabel("Query (Target Tokens)")
    plt.tight_layout()
    plt.show()

def plot_all_heads(attention_weights, sentence_tokens, num_heads=8):
    """绘制所有注意力头的权重"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    attn_weights = attention_weights[0].cpu().numpy()  # 第一个batch
    
    for head in range(num_heads):
        sns.heatmap(attn_weights[head], 
                    xticklabels=sentence_tokens,
                    yticklabels=sentence_tokens,
                    cmap="YlGnBu",
                    ax=axes[head],
                    cbar_kws={'label': 'Weight'})
        axes[head].set_title(f'Head {head + 1}')
    
    plt.tight_layout()
    plt.suptitle("Multi-Head Attention Weights", fontsize=16, y=1.02)
    plt.show()

def main():
    # 设置随机种子以便复现
    torch.manual_seed(41)
    
    # 示例英文句子
    sentence = "The quick brown fox jumps over the lazy dog"
    tokens = sentence.split()
    token_to_id = {token: idx + 1 for idx, token in enumerate(tokens)}  # 0留给padding
    token_to_id['<PAD>'] = 0
    
    # 将句子转换为ID序列
    token_ids = [token_to_id[token] for token in tokens]
    input_tensor = torch.tensor([token_ids])  # 添加batch维度
    
    print("输入句子:", sentence)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    print()
    
    # 模型参数
    vocab_size = len(token_to_id)
    d_model = 512
    num_heads = 8
    
    # 创建模型
    model = SingleLayerTransformer(vocab_size, d_model, num_heads)
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    # 获取注意力权重
    attention_weights = model.get_attention_weights()
    
    if attention_weights is not None:
        print(f"注意力权重形状: {attention_weights.shape}")  # [batch, heads, seq_len, seq_len]
        print()
        
        # 可视化第一个注意力头
        print("可视化第一个注意力头的注意力权重:")
        plot_attention_weights(attention_weights, tokens, head_idx=0)
        
        # 可视化所有注意力头
        print("可视化所有8个注意力头的注意力权重:")
        plot_all_heads(attention_weights, tokens, num_heads)
        
        # 分析不同头的注意力模式
        print("不同注意力头的模式分析:")
        attn_weights = attention_weights[0].cpu().numpy()
        
        for head in range(num_heads):
            weights = attn_weights[head]
            # 计算对角线权重（关注自身）和最大非对角线权重
            self_attention = np.mean(np.diag(weights))
            max_cross_attention = np.max(weights - np.diag(np.diag(weights)))
            
            print(f"头 {head + 1}: 自注意力平均={self_attention:.3f}, 最大交叉注意力={max_cross_attention:.3f}")
    
    # 演示掩码功能
    print("\n" + "="*50)
    print("演示因果掩码（Causal Masking）效果:")
    
    # 创建因果掩码
    seq_len = len(tokens)
    causal_mask = create_look_ahead_mask(seq_len)
    print("因果掩码（0表示被掩码的位置）:")
    print(causal_mask.int().numpy())
    
    # 使用掩码的前向传播
    with torch.no_grad():
        masked_output = model(input_tensor, mask=causal_mask.unsqueeze(0).unsqueeze(1))
    
    masked_attention_weights = model.get_attention_weights()
    
    if masked_attention_weights is not None:
        print("\n应用因果掩码后的第一个注意力头:")
        plot_attention_weights(masked_attention_weights, tokens, head_idx=0, 
                              layer_name="Masked Self-Attention")

if __name__ == "__main__":
    main()