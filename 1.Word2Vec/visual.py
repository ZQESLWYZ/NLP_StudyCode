# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import matplotlib.font_manager as fm
from adjustText import adjust_text  # 用于调整标签位置避免重叠

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 加载训练好的Word2Vec模型
model_path = 'shuihuzhuan_word2vec.model'  # 替换为你的模型路径
model = Word2Vec.load(model_path)

def visualize_word_vectors(words, title="水浒传人物词向量可视化"):
    """
    使用t-SNE对选定词语进行降维可视化
    :param words: 要可视化的词语列表
    :param title: 图表标题
    """
    # 获取词语向量
    vectors = []
    valid_words = []
    
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)
        else:
            print(f"警告: 词语 '{word}' 不在词汇表中，已跳过")
    
    if not vectors:
        print("错误: 没有有效的词语可用于可视化")
        return
    
    vectors = np.array(vectors)
    
    # 使用t-SNE进行降维
    tsne = TSNE(
        n_components=2,      # 降维到2维
        perplexity=15,       # 困惑度（通常5-50）
        early_exaggeration=12.0, # 早期放大因子
        learning_rate=200.0, # 学习率
        n_iter_without_progress=1000,         # 迭代次数
        random_state=42       # 随机种子
    )
    
    print("正在进行t-SNE降维...")
    vectors_2d = tsne.fit_transform(vectors)
    print("降维完成!")
    
    # 创建散点图
    plt.figure(figsize=(15, 12))
    plt.title(title, fontsize=20)
    
    # 绘制散点
    x = vectors_2d[:, 0]
    y = vectors_2d[:, 1]
    plt.scatter(x, y, alpha=0.7, s=60)
    
    # 添加标签
    texts = []
    for i, word in enumerate(valid_words):
        texts.append(plt.text(x[i], y[i], word, fontsize=12))
    
    # 调整标签位置避免重叠
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                expand_points=(1.5, 1.5),  # 扩展点周围的空间
                expand_text=(1.1, 1.1),    # 扩展文本周围的空间
                force_text=(0.5, 0.5))     # 文本移动的力
    
    # 添加网格和坐标轴标签
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("t-SNE维度 1", fontsize=14)
    plt.ylabel("t-SNE维度 2", fontsize=14)
    
    # 保存图像
    plt.savefig('shuihu_word_vectors.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 选择要可视化的词语
    # 这里选取水浒传中的主要人物、地点和概念
    words_to_visualize = [
        # 主要人物
        "宋江", "卢俊义", "吴用", "公孙胜", "关胜", "林冲", "秦明", 
        "呼延灼", "花荣", "柴进", "李应", "朱仝", "鲁智深", "武松", 
        "董平", "张清", "杨志", "徐宁", "索超", "戴宗", "刘唐", 
        "李逵", "史进", "穆弘", "雷横", "李俊", "阮小二", "张横", 
        "阮小五", "张顺", "阮小七", "杨雄", "石秀", "解珍", "解宝", 
        "燕青",
        
        # 女性角色
        "潘金莲", "阎婆惜", "扈三娘", "孙二娘", "顾大嫂",
        
        # 地点
        "梁山", "东京", "江州", "大名府", "沧州", "祝家庄", "曾头市",
        
        # 概念
        "好汉", "朝廷", "招安", "忠义", "造反", "聚义", "替天行道",
        "劫富济贫", "官军", "山寨", "头领", "兄弟", "义气"
    ]
    
    # 执行可视化
    visualize_word_vectors(words_to_visualize, "《水浒传》词向量t-SNE可视化")