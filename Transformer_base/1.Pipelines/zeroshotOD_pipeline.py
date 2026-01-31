import os
import torch
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import *
from PIL import Image

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897' 
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

# 1. 加载模型和处理器
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# 2. 准备图片
# 你可以从本地文件加载
# image = Image.open("path/to/your/image.jpg").convert("RGB")

# 或者从URL加载（示例图片）
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. 文本描述（要检测的对象）
texts = ["remote control", "cat"]  # 要检测的物体描述

# 4. 处理输入
inputs = processor(images=image, text=texts, return_tensors="pt")

# 5. 推理
with torch.no_grad():
    outputs = model(**inputs)
    
# 6. 后处理（GroundingDINO 需要手动后处理）
# 获取预测结果
logits = outputs.logits
boxes = outputs.pred_boxes

# 7. 设置阈值过滤结果
text_threshold = 0.25
box_threshold = 0.3

# 解析结果
scores = torch.sigmoid(logits)  # 转换为概率
probs = scores.max(dim=-1)[0]  # 每个预测框的最大概率
keep = probs > box_threshold  # 过滤低置信度框

# 获取过滤后的结果
filtered_probs = probs[keep]
filtered_boxes = boxes[keep]
filtered_scores = scores[keep]

# 8. 显示结果
print(f"检测到 {len(filtered_boxes)} 个目标:")
for i, (prob, box) in enumerate(zip(filtered_probs, filtered_boxes)):
    # 获取对应的类别
    class_idx = scores[keep][i].argmax()
    class_name = texts[class_idx] if class_idx < len(texts) else f"object_{class_idx}"
    
    print(f"目标 {i+1}:")
    print(f"  类别: {class_name}")
    print(f"  置信度: {prob:.4f}")
    print(f"  边界框: {box.tolist()}")
    print("-" * 40)

# 9. 可视化结果
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for i, (prob, box) in enumerate(zip(filtered_probs, filtered_boxes)):
    # 获取对应的类别
    class_idx = scores[keep][i].argmax()
    class_name = texts[class_idx] if class_idx < len(texts) else f"object_{class_idx}"
    
    # 绘制边界框
    x1, y1, x2, y2 = box.tolist()
    width = x2 - x1
    height = y2 - y1
    
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    # 添加标签
    ax.text(
        x1, y1 - 5,
        f"{class_name}: {prob:.2f}",
        bbox=dict(facecolor='yellow', alpha=0.7),
        fontsize=10
    )

plt.title("GroundingDINO Zero-Shot Object Detection")
plt.axis('off')
plt.show()

