import os
import argparse
import sentencepiece as spm
from tqdm import tqdm

def load_tokenizer(model_path):
    """
    加载 SentencePiece 模型
    :param model_path: .model 文件路径
    :return: SentencePieceProcessor 实例
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    print(f"✅ 成功加载 tokenizer: {model_path}")
    print(f"   词表大小: {sp.get_piece_size()}")
    return sp

def tokenize_file(input_file, output_file, sp, output_ids=False):
    """
    对输入文本文件逐行分词，并写入输出文件
    :param input_file: 输入 .txt 文件路径（一行一句）
    :param output_file: 输出文件路径
    :param sp: 已加载的 SentencePieceProcessor
    :param output_ids: 是否输出 token ID（True）还是 token 字符串（False）
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 自动创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        lines = [line.strip() for line in fin if line.strip()]
        
        for line in tqdm(lines, desc="🔤 分词进度"):
            if output_ids:
                # 输出 token ID 序列（空格分隔）
                ids = sp.encode(line)
                fout.write(' '.join(map(str, ids)) + '\n')
            else:
                # 输出 token 字符串（如 '▁I ▁love ▁NLP'）
                pieces = sp.encode_as_pieces(line)
                fout.write(' '.join(pieces) + '\n')

    print(f"🎉 分词完成！结果已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="使用 SentencePiece 模型对 TXT 文件进行批量分词")
    parser.add_argument('--input', type=str, default=r"0.Tools\DATAtools\result\example_en.txt", help='输入文本文件路径 (.txt)')
    parser.add_argument('--model', type=str,default=r"0.Tools\DATAtools\result\spm_en.model", help='SentencePiece 模型路径 (.model)')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（默认自动生成）')
    parser.add_argument('--ids', action='store_true', help='是否输出 token ID 而非 token 字符串')

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        suffix = "_ids.txt" if args.ids else "_tokens.txt"
        args.output = base_name + suffix

    try:
        # 1. 加载 tokenizer
        sp = load_tokenizer(args.model)

        # 2. 执行分词
        tokenize_file(args.input, args.output, sp, output_ids=args.ids)

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == '__main__':
    main()