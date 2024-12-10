# 构建词汇表
import pandas as pd
from collections import Counter

# Step 1: 读取处理后的文件
file_path = 'clean6.txt'  # 替换为您的文件路径
df = pd.read_csv(file_path, sep='\t', header=None)

# Step 2: 统计词频
# 连接 source 和 target 列的文本，拆分成单词
vocab = Counter(' '.join(df[0]).split() + ' '.join(df[1]).split())

# Step 3: 过滤低频词（频率大于等于 5）
vocab = {k: v for k, v in vocab.items() if v >= 5}

# Step 4: 保存词汇表到 vocab.txt
vocab_file_path = 'clean7.txt'  # 输出文件名
with open(vocab_file_path, 'w', encoding='utf-8') as f:
    for word in vocab:
        f.write(word + '\n')

print(f"词汇表已保存到 {vocab_file_path}")
