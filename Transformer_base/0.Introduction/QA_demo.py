import gradio as gr
from transformers import pipeline
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897' 
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'


# 手动创建问答pipeline
qa_pipeline = pipeline("question-answering", 
                       model="uer/roberta-base-chinese-extractive-qa")

def answer_question(context, question):
    result = qa_pipeline(question=question, context=context)
    return f"答案: {result['answer']}\n置信度: {result['score']:.2%}"

interface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(lines=5, label="上下文", placeholder="输入文章或段落..."),
        gr.Textbox(label="问题", placeholder="输入要提问的问题...")
    ],
    outputs=gr.Textbox(label="答案"),
    title="中文问答系统",
    description="基于RoBERTa的中文抽取式问答"
)

interface.launch()