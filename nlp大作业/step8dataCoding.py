# 进行数据编码
import pandas as pd

# Step 1: 读取处理后的文件
file_path = 'clean6.txt'  # 替换为您的文件路径
df = pd.read_csv(file_path, sep='\t', header=None)

# Step 2: 加载词汇表并创建映射
# 显式指定编码为 utf-8
with open('clean7.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().splitlines()

word2idx = {word: idx for idx, word in enumerate(vocab)}

# 为 <unk> 和 <pad> 添加映射
word2idx['<unk>'] = len(word2idx)  # 未知词
word2idx['<pad>'] = len(word2idx) + 1  # 填充词

# Step 3: 转换句子为 ID
df[0] = df[0].apply(lambda x: [word2idx.get(w, word2idx['<unk>']) for w in x.split()])
df[1] = df[1].apply(lambda x: [word2idx.get(w, word2idx['<unk>']) for w in x.split()])

# Step 4: 填充序列
max_seq_len = 50  # 定义最大序列长度
df[0] = df[0].apply(lambda x: x[:max_seq_len] + [word2idx['<pad>']] * (max_seq_len - len(x)))
df[1] = df[1].apply(lambda x: x[:max_seq_len] + [word2idx['<pad>']] * (max_seq_len - len(x)))

# Step 5: 保存编码后的数据
output_file_path = 'clean8.txt'  # 设置输出文件名
df.to_csv(output_file_path, sep='\t', index=False, header=False)

print(f"数据编码完成，结果已保存到 {output_file_path}")
