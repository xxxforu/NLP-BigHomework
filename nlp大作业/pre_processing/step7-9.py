import pandas as pd
from collections import Counter
import torch
import os

# 读取并处理每个文件的函数
def process_file(file_path, vocab_file_path, output_file_path, max_seq_len=50):
    # Step 1: 读取处理后的文件
    df = pd.read_csv(file_path, sep='\t', header=None)

    # Step 2: 统计词频并生成词汇表
    vocab = Counter(' '.join(df[0]).split() + ' '.join(df[1]).split())

    # Step 3: 过滤低频词（频率大于等于 5）
    vocab = {k: v for k, v in vocab.items() if v >= 5}

    # Step 4: 保存词汇表到 vocab.txt
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')
    print(f"词汇表已保存到 {vocab_file_path}")

    # Step 5: 加载词汇表并创建映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    word2idx['<unk>'] = len(word2idx)  # 未知词
    word2idx['<pad>'] = len(word2idx) + 1  # 填充词

    # Step 6: 转换句子为 ID
    df[0] = df[0].apply(lambda x: [word2idx.get(w, word2idx['<unk>']) for w in x.split()])
    df[1] = df[1].apply(lambda x: [word2idx.get(w, word2idx['<unk>']) for w in x.split()])

    # Step 7: 填充序列
    df[0] = df[0].apply(lambda x: x[:max_seq_len] + [word2idx['<pad>']] * (max_seq_len - len(x)))
    df[1] = df[1].apply(lambda x: x[:max_seq_len] + [word2idx['<pad>']] * (max_seq_len - len(x)))

    # Step 8: 保存编码后的数据
    df.to_csv(output_file_path, sep='\t', index=False, header=False)
    print(f"数据编码完成，结果已保存到 {output_file_path}")

    # Step 9: 保存为 PyTorch Tensor
    torch.save((df[0].tolist(), df[1].tolist()), output_file_path.replace('.txt', '.pt'))
    print(f"数据已保存为 {output_file_path.replace('.txt', '.pt')}")

    # Step 10: 保存为 JSON
    df.to_json(output_file_path.replace('.txt', '.json'), orient='records', lines=True)
    print(f"数据已保存为 {output_file_path.replace('.txt', '.json')}")

    # Step 11: 保存为 CSV
    df.to_csv(output_file_path.replace('.txt', '.csv'), index=False)
    print(f"数据已保存为 {output_file_path.replace('.txt', '.csv')}")


# 批量处理多个文件的函数
def batch_process(input_files, vocab_output_files, output_files):
    for input_file, vocab_output_file, output_file in zip(input_files, vocab_output_files, output_files):
        process_file(input_file, vocab_output_file, output_file)

# 输入文件路径和输出文件路径
input_files = ['test1.txt', 'train1.txt', 'val1.txt']  # 输入文件列表
vocab_output_files = ['test_vocab.txt', 'train_vocab.txt', 'val_vocab.txt']  # 输出词汇表路径列表
output_files = ['test.txt', 'train.txt', 'val.txt']  # 输出处理后文件路径列表

# 批量处理
batch_process(input_files, vocab_output_files, output_files)