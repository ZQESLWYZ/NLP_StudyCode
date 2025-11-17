import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import os
import pickle
import jieba
from collections import Counter
import math

class WordLevelTextProcessor:
    """单词级文本处理器（中文分词）"""
    def __init__(self, file_path, seq_length=20, cache_dir="word_processed_data", min_freq=2):
        self.file_path = file_path
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.words = None
        self.sequences = None
        self.next_words = None
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化jieba分词
        jieba.initialize()
    
    def get_cache_filename(self, suffix):
        """生成缓存文件名"""
        base_name = os.path.basename(self.file_path).replace('.', '_')
        return os.path.join(self.cache_dir, f"{base_name}_{suffix}.pkl")
    
    def load_and_clean_text(self, force_reload=False):
        """加载并清洗中文文本，支持缓存"""
        cache_file = self.get_cache_filename("cleaned_text")
        
        # 如果存在缓存且不强制重新加载，则从缓存读取
        if not force_reload and os.path.exists(cache_file):
            print("从缓存加载已清洗的文本...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        
        print("加载和清洗文本...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 清洗文本：保留中文字符、标点符号和基本符号
        text = re.sub(r'[^\u4e00-\u9fa5，。！？；：""（）《》\s]', '', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # 保存到缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"已保存清洗后的文本到: {cache_file}")
        
        return text
    
    def segment_text(self, text, force_resegment=False):
        """对文本进行分词"""
        cache_file = self.get_cache_filename("segmented_words")
        
        # 如果存在缓存且不强制重新分词，则从缓存读取
        if not force_resegment and os.path.exists(cache_file):
            print("从缓存加载分词结果...")
            with open(cache_file, 'rb') as f:
                words = pickle.load(f)
            self.words = words
            return words
        
        print("对文本进行分词...")
        # 使用jieba进行分词
        words = []
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 分词并过滤空字符串
                seg_list = [word for word in jieba.cut(sentence) if word.strip()]
                words.extend(seg_list)
                # 添加句子结束标记
                words.append('<EOS>')
        
        # 移除最后一个<EOS>（如果有）
        if words and words[-1] == '<EOS>':
            words = words[:-1]
        
        self.words = words
        
        # 保存到缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(words, f)
        print(f"已保存分词结果到: {cache_file}")
        print(f"分词后单词数量: {len(words)}")
        
        return words
    
    def build_vocabulary(self, words=None, force_rebuild=False):
        """构建词汇表，支持缓存"""
        if words is None:
            words = self.words
        
        cache_file = self.get_cache_filename("vocabulary")
        
        # 如果存在缓存且不强制重新构建，则从缓存读取
        if not force_rebuild and os.path.exists(cache_file):
            print("从缓存加载词汇表...")
            with open(cache_file, 'rb') as f:
                vocab_data = pickle.load(f)
                self.word2idx = vocab_data['word2idx']
                self.idx2word = vocab_data['idx2word']
                self.vocab_size = vocab_data['vocab_size']
            print(f"词汇表大小: {self.vocab_size}")
            return self.word2idx, self.idx2word
        
        print("构建词汇表...")
        # 统计词频
        word_counter = Counter(words)
        
        # 过滤低频词
        if self.min_freq > 1:
            word_counter = {word: count for word, count in word_counter.items() if count >= self.min_freq}
            print(f"过滤低频词后，唯一词数: {len(word_counter)}")
        
        # 创建词到索引的映射
        vocab_words = ['<PAD>', '<UNK>', '<START>', '<EOS>'] + sorted(word_counter.keys())
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab_words)
        
        # 保存到缓存
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"词汇表大小: {self.vocab_size}")
        print(f"已保存词汇表到: {cache_file}")
        return self.word2idx, self.idx2word
    
    def words_to_sequences(self, words=None, force_reprocess=False):
        """将单词序列转换为数字序列，支持缓存"""
        if words is None:
            words = self.words
        
        cache_file = self.get_cache_filename("sequences")
        
        # 如果存在缓存且不强制重新处理，则从缓存读取
        if not force_reprocess and os.path.exists(cache_file):
            print("从缓存加载序列数据...")
            with open(cache_file, 'rb') as f:
                sequence_data = pickle.load(f)
                self.sequences = sequence_data['sequences']
                self.next_words = sequence_data['next_words']
            print(f"序列数量: {len(self.sequences)}")
            return self.sequences, self.next_words
        
        print("转换为序列...")
        sequences = []
        next_words = []
        
        for i in range(0, len(words) - self.seq_length, 1):
            seq = words[i:i + self.seq_length]
            next_word = words[i + self.seq_length]
            
            # 将单词转换为索引
            seq_indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in seq]
            next_word_idx = self.word2idx.get(next_word, self.word2idx['<UNK>'])
            
            sequences.append(seq_indices)
            next_words.append(next_word_idx)
        
        self.sequences = np.array(sequences)
        self.next_words = np.array(next_words)
        
        # 保存到缓存
        sequence_data = {
            'sequences': self.sequences,
            'next_words': self.next_words
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(sequence_data, f)
        
        print(f"序列数量: {len(sequences)}")
        print(f"已保存序列数据到: {cache_file}")
        return self.sequences, self.next_words
    
    def get_processed_data(self, force_reload=False):
        """获取所有处理后的数据，支持缓存"""
        # 如果已经处理过且不强制重新加载，直接返回
        if not force_reload and self.sequences is not None and self.next_words is not None:
            return self.sequences, self.next_words
        
        # 按顺序处理数据
        text = self.load_and_clean_text(force_reload)
        words = self.segment_text(text, force_reload)
        self.build_vocabulary(words, force_reload)
        return self.words_to_sequences(words, force_reload)
    
    def save_processing_info(self, info_file="word_processing_info.txt"):
        """保存处理信息到文件"""
        info = {
            'file_path': self.file_path,
            'seq_length': self.seq_length,
            'vocab_size': self.vocab_size,
            'text_length': len(self.words) if self.words else 0,
            'num_sequences': len(self.sequences) if self.sequences is not None else 0,
            'min_freq': self.min_freq
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"处理信息已保存到: {info_file}")
        return info

class WordLevelRNN(nn.Module):
    """单词级RNN语言模型"""
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=512, 
                 n_layers=2, dropout=0.3):
        super(WordLevelRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # 词嵌入
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
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

class WordLevelTransformer(nn.Module):
    """单词级Transformer语言模型"""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=512, max_seq_len=100, dropout=0.1):
        super(WordLevelTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 输出层
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_positional_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # 词嵌入 + 位置编码
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            x = x + self.pos_encoding[:, :self.max_seq_len, :]
        
        x = self.dropout(x)
        
        # 创建因果掩码（用于生成任务）
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            mask = mask.to(x.device)
        
        # Transformer编码器
        x = self.transformer(x, mask=mask)
        
        # 输出层
        x = self.layer_norm(x)
        logits = self.output_layer(x)
        
        return logits
    
    def generate(self, start_words, max_length=20, temperature=1.0, top_k=None):
        """生成文本"""
        self.eval()
        generated = start_words.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                if generated.size(1) > self.max_seq_len:
                    input_seq = generated[:, -self.max_seq_len:]
                else:
                    input_seq = generated
                
                logits = self.forward(input_seq)
                next_word_logits = logits[:, -1, :] / temperature
                
                # Top-k采样
                if top_k is not None:
                    indices_to_remove = next_word_logits < torch.topk(next_word_logits, top_k)[0][..., -1, None]
                    next_word_logits[indices_to_remove] = -float('inf')
                
                # 采样下一个词
                probs = torch.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成的序列
                generated = torch.cat([generated, next_word], dim=1)
        
        return generated

class WordLevelTextGenerator:
    """单词级文本生成器"""
    def __init__(self, model, word2idx, idx2word, device):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.model.eval()
        self.model.to(device)
    
    def generate_text(self, start_text, length=20, temperature=1.0, top_k=None):
        """生成文本"""
        self.model.eval()
        
        # 分词
        words = [word for word in jieba.cut(start_text) if word.strip()]
        
        # 转换为索引
        word_indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # 确保序列长度不超过模型最大长度
        max_seq_len = getattr(self.model, 'max_seq_len', 50)
        if len(word_indices) > max_seq_len:
            word_indices = word_indices[-max_seq_len:]
        
        input_seq = torch.tensor([word_indices], dtype=torch.long).to(self.device)
        
        # 生成文本
        if isinstance(self.model, WordLevelTransformer):
            generated = self.model.generate(input_seq, max_length=length, 
                                          temperature=temperature, top_k=top_k)
            generated_indices = generated[0].tolist()
        else:
            # RNN生成
            generated_indices = word_indices.copy()
            self.model.eval()
            
            with torch.no_grad():
                hidden = self.model.init_hidden(1, self.device) if hasattr(self.model, 'init_hidden') else None
                current_input = input_seq
                
                for _ in range(length):
                    # 前向传播
                    output, hidden = self.model(current_input, hidden)
                    
                    # 应用温度参数
                    output = output / temperature
                    
                    # Top-k采样
                    if top_k is not None:
                        indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
                        output[indices_to_remove] = -float('inf')
                    
                    probabilities = torch.softmax(output, dim=1)
                    
                    # 采样下一个词
                    predicted = torch.multinomial(probabilities, 1).item()
                    generated_indices.append(predicted)
                    
                    # 更新输入
                    current_input = torch.tensor([[predicted]], dtype=torch.long).to(self.device)
        
        # 将索引转换回单词
        generated_words = [self.idx2word[idx] for idx in generated_indices]
        
        # 组合成文本
        generated_text = ''.join(generated_words)
        
        return generated_text

class WordLevelLMTrainer:
    """单词级语言模型训练器"""
    def __init__(self, model, processor, device=None):
        self.model = model
        self.processor = processor
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"使用设备: {self.device}")
    
    def train(self, X, y, epochs=50, batch_size=4096, learning_rate=1e-3, model_type='rnn'):
        """训练模型"""
        # 将数据转换为PyTorch张量并移动到设备
        X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                # 数据已经在设备上，无需再次移动
                optimizer.zero_grad()
                
                # 前向传播
                if model_type == 'transformer':
                    logits = self.model(batch_x)
                    # Transformer输出所有位置的预测，我们只取最后一个位置
                    if len(logits.shape) == 3:
                        logits = logits[:, -1, :]
                else:
                    logits, _ = self.model(batch_x)
                
                # 计算损失
                loss = criterion(logits, batch_y)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
                # 生成示例文本
                try:
                    generator = WordLevelTextGenerator(self.model, self.processor.word2idx, 
                                                     self.processor.idx2word, self.device)
                    sample_text = generator.generate_text("今天天气", length=10, temperature=0.8)
                    print(f"生成示例: {sample_text}")
                except Exception as e:
                    print(f"生成示例文本时出错: {e}")
        
        return self.model

# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化处理器
    processor = WordLevelTextProcessor(
        "2.LM/dataset/chinese_novel.txt",  # 替换为您的文件路径
        seq_length=20,
        min_freq=2
    )
    
    # 获取处理后的数据
    sequences, next_words = processor.get_processed_data()
    
    # 选择模型类型：'rnn' 或 'transformer'
    model_type = 'rnn'  # 可以改为 'transformer'
    
    if model_type == 'rnn':
        # 创建RNN模型
        model = WordLevelRNN(
            vocab_size=processor.vocab_size,
            embedding_dim=200,
            hidden_dim=512,
            n_layers=2,
            dropout=0.3
        )
    else:
        # 创建Transformer模型
        model = WordLevelTransformer(
            vocab_size=processor.vocab_size,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=512,
            max_seq_len=50,
            dropout=0.1
        )
    
    # 训练模型
    trainer = WordLevelLMTrainer(model, processor, device)
    model = trainer.train(sequences, next_words, epochs=50, batch_size=2048, 
                         learning_rate=1e-3, model_type=model_type)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor_data': {
            'word2idx': processor.word2idx,
            'idx2word': processor.idx2word,
            'vocab_size': processor.vocab_size
        },
        'model_type': model_type,
        'model_config': {
            'embedding_dim': 200 if model_type == 'rnn' else 256,
            'hidden_dim': 512 if model_type == 'rnn' else None,
            'n_layers': 2 if model_type == 'rnn' else 6,
            'd_model': 256 if model_type == 'transformer' else None,
            'n_heads': 8 if model_type == 'transformer' else None,
            'd_ff': 512 if model_type == 'transformer' else None,
        }
    }, "word_level_lm_model.pth")
    
    print("模型训练完成并已保存！")
    
    # 加载模型进行文本生成
    checkpoint = torch.load("word_level_lm_model.pth", map_location=device)
    
    if checkpoint['model_type'] == 'rnn':
        loaded_model = WordLevelRNN(checkpoint['processor_data']['vocab_size'])
    else:
        loaded_model = WordLevelTransformer(
            checkpoint['processor_data']['vocab_size'],
            max_seq_len=50
        )
    
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.to(device)
    
    generator = WordLevelTextGenerator(
        loaded_model, 
        checkpoint['processor_data']['word2idx'], 
        checkpoint['processor_data']['idx2word'],
        device
    )
    
    # 生成文本示例
    generated_text = generator.generate_text("春天来了", length=20, temperature=0.7)
    print("生成的文本:")
    print(generated_text)