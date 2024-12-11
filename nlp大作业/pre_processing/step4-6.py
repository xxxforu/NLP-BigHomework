import jieba
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
import pandas as pd

# 下载nltk的punkt标记化工具（如未下载，可以取消注释运行）
# nltk.download('punkt')

# 加载BERT子词分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义处理函数
def process_file(input_file, output_file):
    # 存储有效行
    valid_lines = []

    # 读取和预处理数据
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            # 跳过空行
            if not line.strip():
                continue

            # 尝试分割每行数据为英文和中文句子
            parts = line.strip().split('\t')

            # 如果分割结果不是两个部分，标记该行为无效
            if len(parts) != 2:
                print(f"删除无效行：{line.strip()}")
                continue

            en_sentence, zh_sentence = parts

            # 对英文句子进行标记化
            en_tokens = word_tokenize(en_sentence)  # 使用nltk进行英文分词

            # 对中文句子进行分词
            zh_tokens = jieba.cut(zh_sentence)  # 使用jieba进行中文分词

            # 将分词结果转换为以空格分隔的字符串
            en_processed = ' '.join(en_tokens)
            zh_processed = ' '.join(zh_tokens)

            # 存储有效的处理结果
            valid_lines.append(f"{en_processed}\t{zh_processed}\n")

    # 对英文句子进行子词切分
    processed_lines = []
    for line in valid_lines:
        if "\t" in line:
            english_text, chinese_text = line.split("\t")

            # 英文标记化后的子词切分
            tokens = word_tokenize(english_text)
            subwords = tokenizer.tokenize(" ".join(tokens))

            # 重组文本并添加特殊标记
            processed_line = f"<sos> {' '.join(subwords)} <eos>\t<sos> {chinese_text.strip()} <eos>\n"
            processed_lines.append(processed_line)

    # 保存处理后的结果
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(processed_lines)

    print(f"处理完成，结果已保存到 {output_file}")

# 批量处理多个文件
def batch_process(input_files, output_files):
    for input_file, output_file in zip(input_files, output_files):
        process_file(input_file, output_file)

# 输入文件路径和输出文件路径
input_files = ['test.txt', 'train.txt', 'val.txt']  # 输入文件列表
output_files = ['test1.txt', 'train1.txt', 'val1.txt']  # 输出文件列表

# 批量处理
batch_process(input_files, output_files)
