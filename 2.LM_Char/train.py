import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter
import torch.optim as optim
from model.rnn import ChineseRNN, TextGenerator
from dataprocess import ChineseTextProcessor

class ChineseNovelTrainer:
    def __init__(self, file_path, seq_length=50, batch_size=64):
        self.processor = ChineseTextProcessor(file_path, seq_length)
        self.batch_size = batch_size
        
    def prepare_data(self, save_path=None):
        """准备训练数据，可选择保存中间数据
        Args:
            save_path: 保存中间数据的路径，如果为None则不保存
        """
        print("加载和清洗文本...")
        text = self.processor.load_and_clean_text()
        
        print("构建词汇表...")
        self.processor.build_vocabulary(text)
        
        print("转换为序列...")
        X, y = self.processor.text_to_sequences(text)
        
        # 转换为PyTorch张量
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        
        # 保存中间数据
        if save_path:
            print(f"保存中间数据到 {save_path}")
            torch.save({
                'X': X,
                'y': y,
                'char2idx': self.processor.char2idx,
                'idx2char': self.processor.idx2char,
                'vocab_size': self.processor.vocab_size
            }, save_path)
        
        print(f"训练样本数: {len(X)}")
        return self.X, self.y
        
    def load_intermediate_data(self, path):
        """加载已保存的中间数据
        Args:
            path: 中间数据文件路径
        """
        print(f"加载中间数据从 {path}")
        data = torch.load(path)
        self.X = torch.tensor(data['X'], dtype=torch.long)
        self.y = torch.tensor(data['y'], dtype=torch.long)
        self.processor.char2idx = data['char2idx']
        self.processor.idx2char = data['idx2char']
        self.processor.vocab_size = data['vocab_size']
        print(f"训练样本数: {len(self.X)}")
        return self.X, self.y
    
    def train_model(self, epochs=50, learning_rate=0.001):
        """训练模型"""
        # 准备数据
        X, y = self.prepare_data()
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将数据移动到设备上
        X = X.to(device)
        y = y.to(device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                shuffle=True)
        
        # 初始化模型
        model = ChineseRNN(self.processor.vocab_size).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print("开始训练...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                output, _ = model(batch_x)
                loss = criterion(output, batch_y)
                
                # 反向传播
                loss.backward()
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
                # 每5个epoch生成示例文本
                generator = TextGenerator(model, self.processor.char2idx, 
                                        self.processor.idx2char)
                sample_text = generator.generate_text("从前", length=50, temperature=0.8, device=device)
                print(f"生成示例: {sample_text}")
        
        self.model = model
        return model
    
    def save_model(self, path):
        """保存模型和词汇表"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char2idx': self.processor.char2idx,
            'idx2char': self.processor.idx2char,
            'vocab_size': self.processor.vocab_size
        }, path)
        print(f"模型已保存到: {path}")

# 使用示例
if __name__ == "__main__":
    # 初始化训练器
    trainer = ChineseNovelTrainer("2.LM\dataset\chinese_novel.txt", seq_length=50, batch_size=128)
    
    # 训练模型
    model = trainer.train_model(epochs=100, learning_rate=0.001)
    
    # 保存模型
    trainer.save_model("chinese_rnn_model.pth")
    
    # 加载模型进行文本生成
    checkpoint = torch.load("chinese_rnn_model.pth")
    loaded_model = ChineseRNN(checkpoint['vocab_size'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    generator = TextGenerator(loaded_model, checkpoint['char2idx'], checkpoint['idx2char'])
    
    # 生成文本示例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = loaded_model.to(device)
    generated_text = generator.generate_text("春天", length=200, temperature=0.7, device=device)
    print("生成的文本:")
    print(generated_text)