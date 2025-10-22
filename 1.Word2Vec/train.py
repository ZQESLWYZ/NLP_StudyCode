# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 步骤1：数据预处理
def preprocess_text(input_file, output_file):
    """
    预处理文本：分词、清洗并保存为分词后的文本
    :param input_file: 原始文本路径
    :param output_file: 分词后输出路径
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # 移除换行符和首尾空格
            line = line.strip()
            if not line:
                continue
            
            # 使用jieba进行分词
            words = jieba.cut(line)
            # 过滤单字和标点符号（根据需求调整）
            filtered_words = [
                word for word in words 
                if len(word) > 1 and not word.isspace()  # 保留长度>1的词语
            ]
            # 将分词结果写入新文件（每行一个句子）
            f_out.write(" ".join(filtered_words) + "\n")
            

# 步骤2：训练Word2Vec模型
def train_word2vec(corpus_file, model_save_path):
    """
    训练Word2Vec模型并保存
    :param corpus_file: 分词后的语料库文件路径
    :param model_save_path: 模型保存路径
    """
    # 将文本转换为LineSentence格式（每行=一个句子，词语已空格分隔）
    sentences = LineSentence(corpus_file)
    
    # 配置模型参数
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,     # 词向量维度
        window=5,            # 上下文窗口大小
        min_count=5,         # 忽略出现次数<5的词语
        workers=4,           # 并行线程数
        epochs=10,           # 训练轮次
        sg=0,                # 1=skip-gram, 0=CBOW
        hs=0,                # 0=负采样, 1=分层softmax
        negative=5,          # 负采样数量
        seed=42              # 随机种子
    )
    
    # 保存模型
    model.save(model_save_path)
    print(f"模型已保存至 {model_save_path}")
    
# 步骤3：使用模型示例
def test_model(model_path):
    """加载模型并测试相似词"""
    model = Word2Vec.load(model_path)
    
    # 示例1：查找相似词
    print("与'女子'最相似的词：")
    for word, sim in model.wv.most_similar('女子', topn=5):
        print(f"{word}: {sim:.4f}")
    
    print(model.wv["好汉"])
    
if __name__ == "__main__":

    # 文件路径配置
    original_text = 'shuihuzhuan.txt'          # 原始水浒传文本
    processed_text = 'shuihuzhuan_processed.txt' # 预处理后文本
    model_path = 'shuihuzhuan_word2vec.model'    # 模型保存路径
    
    # 执行预处理
    preprocess_text(original_text, processed_text)
    print("文本预处理完成")
    
    train_word2vec(processed_text, model_path)
    
    test_model(model_path)