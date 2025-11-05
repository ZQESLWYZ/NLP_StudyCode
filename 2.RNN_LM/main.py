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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据预处理类（中文版）
class ChineseTextProcessor:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def load_text_from_file(self, file_path):
        """从文件加载中文文本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except UnicodeDecodeError:
            # 如果utf-8失败，尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                text = f.read()
            return text
        
    def preprocess_text(self, text):
        """中文文本预处理"""
        # 去除特殊字符，保留中文、英文和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？；：""''、（）《》]', '', text)
        
        # 使用jieba进行中文分词
        sentences = re.split(r'[。！？]', text)
        processed_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # 分词
                words = jieba.lcut(sentence.strip())
                if words:  # 确保分词后不为空
                    # 添加句子标记
                    processed_sentences.append(['<sos>'] + words + ['<eos>'])
        
        return processed_sentences
    
    def build_vocab(self, sentences):
        """构建中文词汇表"""
        word_count = Counter()
        for sentence in sentences:
            word_count.update(sentence)
        
        # 过滤低频词
        vocab = ['<unk>', '<sos>', '<eos>', '<pad>']  # 特殊标记
        vocab += [word for word, count in word_count.items() 
                 if count >= self.min_freq and word not in vocab]
        
        self.vocab_size = len(vocab)
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
    def text_to_sequences(self, sentences, seq_length=20):
        """将中文文本转换为序列"""
        sequences = []
        for sentence in sentences:
            # 截断或填充句子
            if len(sentence) < seq_length + 1:
                sentence = sentence + ['<pad>'] * (seq_length + 1 - len(sentence))
            else:
                sentence = sentence[:seq_length + 1]
            
            indices = [self.word2idx.get(word, self.word2idx['<unk>']) 
                      for word in sentence]
            
            # 创建训练样本
            seq = indices[:-1]
            target = indices[1:]
            sequences.append((seq, target))
                
        return sequences

# 2. 自定义数据集类
class LanguageModelDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# 3. RNN语言模型（适用于中文）
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
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# 4. 训练器类（GPU优化版）
class LanguageModelTrainer:
    def __init__(self, model, train_loader, val_loader, word2idx, idx2word, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.word2idx['<pad>'])
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        
        # 将模型移动到GPU
        self.model.to(device)
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # 将数据移动到GPU
            data, targets = data.to(device), targets.to(device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output, _ = self.model(data)
            
            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), 
                                targets.reshape(-1))
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0 and len(self.train_loader) > 10:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                # 将数据移动到GPU
                data, targets = data.to(device), targets.to(device)
                
                output, _ = self.model(data)
                loss = self.criterion(output.reshape(-1, output.size(-1)), 
                                    targets.reshape(-1))
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, epochs, save_path=None):
        """完整训练过程"""
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 每5个epoch生成一个示例文本
            if (epoch + 1) % 5 == 0:
                # 创建文本生成器，确保使用正确的设备
                generator = TextGenerator(self.model, self.word2idx, self.idx2word, device)
                generated = generator.generate_text("今天 天气", max_length=10, temperature=0.8)
                print(f'生成文本: {generated}')
            
            print('-' * 50)
            
        # 训练完成后保存模型
        if save_path:
            self.save_model(save_path)
            
        return train_losses, val_losses
    
    def save_model(self, save_path):
        """保存模型和词汇表"""
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态和词汇表
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'model_params': {
                'vocab_size': self.model.embedding.num_embeddings,
                'embedding_dim': self.model.embedding.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"模型已保存到 {save_path}")

# 5. 文本生成类（修复设备问题）
class TextGenerator:
    def __init__(self, model, word2idx, idx2word, device='cpu'):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.model.eval()
    
    def generate_text(self, start_text, max_length=20, temperature=1.0):
        """生成中文文本"""
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
                ]).to(self.device)  # 确保输入在正确的设备上
                
                # 模型预测
                output, hidden = self.model(input_seq, hidden)
                
                # 获取最后一个词的logits
                logits = output[0, -1] / temperature
                probabilities = torch.softmax(logits, dim=-1)
                
                # 采样下一个词
                next_word_idx = torch.multinomial(probabilities, 1).item()
                next_word = self.idx2word[next_word_idx]
                
                if next_word == '<eos>' or next_word == '<pad>':
                    break
                    
                generated.append(next_word)
                words.append(next_word)
                
        # 将分词结果连接成文本
        return ''.join(generated)  # 中文不需要空格分隔

# 6. 模型加载函数
def load_model(model_path, model_class=RNNLanguageModel, device='cpu'):
    """加载已训练的中文模型"""
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

# 7. 主函数
def main():
    # 文件路径设置
    text_file_path = r"2.RNN_LM\text.txt"  # 替换为你的中文txt文件路径
    model_save_path = "models/chinese_rnn_language_model.pth"  # 模型保存路径
    
    # 数据预处理
    print("正在加载和预处理中文文本...")
    processor = ChineseTextProcessor(min_freq=2)  # 提高最小词频以减少词汇表大小
    
    # 从文件加载文本
    if os.path.exists(text_file_path):
        text = processor.load_text_from_file(text_file_path)
        print(f"已加载文本，长度: {len(text)} 字符")
    else:
        # 如果文件不存在，使用默认示例中文文本
        print(f"文本文件 {text_file_path} 未找到。使用示例文本。")
        text = """
        今天天气很好，阳光明媚。我决定去公园散步。
        公园里有很多人在锻炼身体。孩子们在草地上玩耍。
        我看到一只小鸟在树上唱歌。这真是一个愉快的早晨。
        自然语言处理是人工智能的一个重要分支。
        深度学习模型可以处理各种复杂的任务。
        循环神经网络适合处理序列数据。
        长短期记忆网络能够记忆长期依赖关系。
        中国的文化历史悠久，博大精深。
        北京是中国的首都，拥有丰富的历史遗迹。
        上海是一座现代化的国际大都市。
        广州以美食闻名，吸引了众多游客。
        深圳是科技创新的重要基地。
        """
    
    sentences = processor.preprocess_text(text)
    print(f"分词后句子数量: {len(sentences)}")
    
    processor.build_vocab(sentences)
    sequences = processor.text_to_sequences(sentences, seq_length=15)
    
    print(f"词汇表大小: {processor.vocab_size}")
    print(f"训练序列数量: {len(sequences)}")
    
    # 划分训练集和验证集
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # 创建数据加载器
    train_dataset = LanguageModelDataset(train_sequences)
    val_dataset = LanguageModelDataset(val_sequences)
    
    # 增加批处理大小，充分利用GPU
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, pin_memory=True)
    
    # 初始化模型
    vocab_size = processor.vocab_size
    model = RNNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练模型
    trainer = LanguageModelTrainer(model, train_loader, val_loader, 
                                  processor.word2idx, processor.idx2word, 
                                  learning_rate=0.001)
    train_losses, val_losses = trainer.train(epochs=50, save_path=model_save_path)
    
    # 文本生成示例
    print("\n" + "="*50)
    print("中文文本生成示例:")
    print("="*50)
    
    # 创建文本生成器（使用CPU进行推理，避免GPU内存问题）
    generator = TextGenerator(model, processor.word2idx, processor.idx2word, device='cpu')
    
    # 生成多个示例
    start_texts = [
        "今天天气",
        "自然语言处理",
        "深度学习",
        "北京是"
    ]
    
    for start_text in start_texts:
        generated = generator.generate_text(start_text, max_length=15, temperature=0.7)
        print(f"起始: '{start_text}' -> 生成: '{generated}'")
    
    # 计算困惑度
    def calculate_perplexity(model, data_loader, pad_idx):
        model.eval()
        total_loss = 0
        total_words = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                # 将数据移动到模型所在的设备
                data, targets = data.to(device), targets.to(device)
                
                output, _ = model(data)
                loss = nn.CrossEntropyLoss(ignore_index=pad_idx)(
                    output.reshape(-1, output.size(-1)), 
                    targets.reshape(-1)
                )
                total_loss += loss.item() * targets.numel()
                total_words += targets.numel()
        
        avg_loss = total_loss / total_words
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()
    
    pad_idx = processor.word2idx['<pad>']
    perplexity = calculate_perplexity(model, val_loader, pad_idx)
    print(f"\n验证集困惑度: {perplexity:.2f}")

# 8. 模型使用函数（用于加载已保存的中文模型）
def use_saved_model(model_path, start_texts=None):
    """使用已保存的中文模型生成文本"""
    if start_texts is None:
        start_texts = [
            "今天天气",
            "自然语言处理",
            "深度学习",
            "北京是"
        ]
    
    # 加载模型到CPU
    model, word2idx, idx2word = load_model(model_path, device='cpu')
    
    # 创建文本生成器
    generator = TextGenerator(model, word2idx, idx2word, device='cpu')
    
    # 生成文本
    print("使用已保存模型生成中文文本:")
    print("="*50)
    
    for start_text in start_texts:
        generated = generator.generate_text(start_text, max_length=15, temperature=0.7)
        print(f"起始: '{start_text}' -> 生成: '{generated}'")

if __name__ == "__main__":
    # 安装jieba分词库（如果尚未安装）
    try:
        import jieba
    except ImportError:
        print("请先安装jieba分词库: pip install jieba")
        exit(1)
    
    # 训练新模型
    main()
    
    # 或者使用已保存的模型（取消注释以下行）
    # use_saved_model("models/chinese_rnn_language_model.pth")