import json
import html  # 新增

if __name__ == "__main__":
    files = ['train', 'dev', 'test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = json.load(open('5.Transformer/data/json/' + file + '.json', 'r', encoding="utf-8"))
        for item in corpus:
            # 检查格式
            if not isinstance(item, list) or len(item) != 2:
                continue
            # HTML 实体解码
            en_raw = html.unescape(item[0])
            ch_raw = html.unescape(item[1])
            # 去除首尾空白，并去掉行首多余空格
            en = en_raw.strip()
            ch = ch_raw.lstrip().strip()
            # 过滤空句子和过短句子
            if not en or not ch:
                continue
            if len(ch) < 5 or len(en) < 5:
                continue
            en_lines.append(en + '\n')
            ch_lines.append(ch + '\n')

    with open(ch_path, "w", encoding="utf-8") as fch:
        fch.writelines(ch_lines)
    with open(en_path, "w", encoding="utf-8") as fen:
        fen.writelines(en_lines)

    print("lines of Chinese: ", len(ch_lines))
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")