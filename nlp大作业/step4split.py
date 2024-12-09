import jieba
import nltk
from nltk.tokenize import word_tokenize

# 下载nltk的punkt标记化工具（如未下载，可以取消注释运行）
# nltk.download('punkt')

# 读取数据集
input_file = 'clean3.txt'    # 数据集路径
output_file = 'clean4.txt'   # 处理后的文件路径

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

# 将所有有效行写入输出文件
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.writelines(valid_lines)

print(f"数据预处理完成，处理后的数据保存到 {output_file}")
