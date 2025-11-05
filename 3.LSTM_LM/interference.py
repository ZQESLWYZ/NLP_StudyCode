# inference.py
# 中文RNN语言模型推理脚本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
import random
import os
import jieba  # 中文分词库
import argparse

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. RNN语言模型（与训练时相同）
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(RNNLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN层 - 使用LSTM更适合中文
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # 确保输入在正确的设备上
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
            
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN前向传播
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            # 确保隐藏状态在正确的设备上
            if hidden[0].device != next(self.parameters()).device:
                hidden = (hidden[0].to(next(self.parameters()).device), 
                         hidden[1].to(next(self.parameters()).device))
            output, hidden = self.rnn(embedded, hidden)
        
        # 全连接层
        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden

# 2. 文本生成类
class TextGenerator:
    def __init__(self, model, word2idx, idx2word, device='cpu'):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.model.eval()
        
        # 将模型移动到指定设备
        self.model.to(device)
    
    def generate_text(self, start_text, max_length=20, temperature=1.0, top_k=0, top_p=0.9):
        """生成中文文本
        
        参数:
            start_text: 起始文本
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性 (值越大越随机)
            top_k: 只考虑概率最高的k个词 (0表示禁用)
            top_p: 核采样参数，只考虑累积概率达到p的词
        """
        # 对起始文本进行分词
        words = jieba.lcut(start_text) if start_text else []
        generated = words.copy()
        
        # 初始化隐藏状态
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # 将当前词转换为索引
                input_words = words[-5:]  # 使用最后5个词
                input_seq = torch.tensor([
                    [self.word2idx.get(word, self.word2idx['<unk>']) for word in input_words]
                ]).to(self.device)
                
                # 模型预测
                output, hidden = self.model(input_seq, hidden)
                
                # 获取最后一个词的logits
                logits = output[0, -1] / temperature
                
                # 应用top-k过滤
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # 应用核采样 (top-p)
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的标记
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的标记
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probabilities = torch.softmax(logits, dim=-1)
                
                # 采样下一个词
                next_word_idx = torch.multinomial(probabilities, 1).item()
                next_word = self.idx2word[next_word_idx]
                
                # 如果遇到结束标记，停止生成
                if next_word == '<eos>' or next_word == '<pad>':
                    break
                    
                generated.append(next_word)
                words.append(next_word)
                
        # 将分词结果连接成文本
        return ''.join(generated)  # 中文不需要空格分隔
    
    def generate_multiple(self, start_texts, max_length=20, temperature=1.0, top_k=0, top_p=0.9):
        """批量生成多个文本"""
        results = []
        for start_text in start_texts:
            generated = self.generate_text(start_text, max_length, temperature, top_k, top_p)
            results.append((start_text, generated))
        return results

# 3. 模型加载函数
def load_model(model_path, model_class=RNNLanguageModel, device='cpu'):
    """加载已训练的中文模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取模型参数
        model_params = checkpoint['model_params']
        
        # 创建模型实例
        model = model_class(
            vocab_size=model_params['vocab_size'],
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers']
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)  # 将模型移动到指定设备
        
        # 获取词汇表
        word2idx = checkpoint['word2idx']
        idx2word = checkpoint['idx2word']
        
        print(f"模型已从 {model_path} 加载到 {device}")
        return model, word2idx, idx2word
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None, None

# 4. 主推理函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='中文RNN语言模型推理')
    parser.add_argument('--model_path', type=str, default='models/chinese_rnn_language_model.pth',
                       help='训练好的模型路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='推理设备 (cpu 或 cuda)')
    parser.add_argument('--max_length', type=int, default=20,
                       help='生成文本的最大长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数，控制随机性 (0.1-1.0)')
    parser.add_argument('--top_k', type=int, default=0,
                       help='只考虑概率最高的k个词 (0表示禁用)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='核采样参数，只考虑累积概率达到p的词')
    parser.add_argument('--input_file', type=str, default=None,
                       help='包含起始文本的文件路径 (每行一个起始文本)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='保存生成结果的文件路径')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行推理")
        args.device = 'cpu'
    
    # 加载模型
    model, word2idx, idx2word = load_model(args.model_path, device=args.device)
    if model is None:
        return
    
    # 创建文本生成器
    generator = TextGenerator(model, word2idx, idx2word, device=args.device)
    
    # 准备起始文本
    if args.input_file and os.path.exists(args.input_file):
        # 从文件读取起始文本
        with open(args.input_file, 'r', encoding='utf-8') as f:
            start_texts = [line.strip() for line in f if line.strip()]
    else:
        # 使用默认起始文本
        start_texts = [
            "今天天气",
            "自然语言处理",
            "深度学习",
            "北京是",
            "人工智能",
            "机器学习",
            "神经网络",
            "大数据",
            "云计算",
            "物联网"
        ]
    
    # 生成文本
    print("\n" + "="*60)
    print("中文RNN语言模型文本生成")
    print("="*60)
    print(f"模型: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"最大生成长度: {args.max_length}")
    print(f"温度: {args.temperature}")
    print(f"Top-k: {args.top_k if args.top_k > 0 else '禁用'}")
    print(f"Top-p: {args.top_p if args.top_p > 0 else '禁用'}")
    print("="*60)
    
    results = generator.generate_multiple(
        start_texts, 
        max_length=args.max_length, 
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # 显示结果
    output_lines = []
    for i, (start_text, generated) in enumerate(results, 1):
        output_line = f"{i:2d}. 起始: '{start_text}' -> 生成: '{generated}'"
        print(output_line)
        output_lines.append(output_line)
    
    # 保存结果到文件
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("中文RNN语言模型文本生成结果\n")
            f.write("="*60 + "\n")
            f.write(f"模型: {args.model_path}\n")
            f.write(f"参数: 长度={args.max_length}, 温度={args.temperature}, top-k={args.top_k}, top-p={args.top_p}\n")
            f.write("="*60 + "\n")
            for line in output_lines:
                f.write(line + "\n")
        print(f"\n结果已保存到: {args.output_file}")
    
    # 交互式生成
    print("\n" + "="*60)
    print("交互式文本生成 (输入'quit'退出)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n请输入起始文本: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
                
            if user_input:
                generated = generator.generate_text(
                    user_input, 
                    max_length=args.max_length, 
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                print(f"生成结果: {generated}")
                
        except KeyboardInterrupt:
            print("\n\n程序已退出")
            break
        except Exception as e:
            print(f"生成文本时出错: {e}")

# 5. 批量生成函数
def batch_generate(model_path, start_texts, max_length=20, temperature=0.8, top_k=0, top_p=0.9, device='cpu'):
    """批量生成文本的函数，可供其他脚本调用"""
    model, word2idx, idx2word = load_model(model_path, device=device)
    if model is None:
        return []
    
    generator = TextGenerator(model, word2idx, idx2word, device=device)
    results = generator.generate_multiple(
        start_texts, 
        max_length=max_length, 
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    return results

# 6. 示例使用函数
def example_usage():
    """示例使用函数"""
    model_path = r"2.RNN_LM\RNN_luxun.pth"
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        return
    
    # 示例起始文本
    start_texts = [
        "且看"
    ]
    
    # 批量生成
    results = batch_generate(
        model_path=model_path,
        start_texts=start_texts,
        max_length=30,
        temperature=0.7,
        device='cpu'
    )
    
    print("批量生成示例:")
    for start_text, generated in results:
        print(f"起始: '{start_text}' -> 生成: '{generated}'")

if __name__ == "__main__":
    # 安装jieba分词库（如果尚未安装）
    try:
        import jieba
    except ImportError:
        print("请先安装jieba分词库: pip install jieba")
        exit(1)
    
    example_usage()