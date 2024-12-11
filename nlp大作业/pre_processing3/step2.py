import nltk
import jieba
from collections import Counter
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# 确保 nltk 的 punkt 分词器已下载
#nltk.download('punkt')

# 定义英文和中文的分词器
from nltk.tokenize import word_tokenize


def tokenizer_en(text):
    return word_tokenize(text)


# 中文分词器：使用 jieba 进行分词
def tokenizer_zh(text):
    return list(jieba.cut(text))


# 构建词汇表函数（替代 torchtext）
def build_vocab(sentences, tokenizer, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)

    # 构建词汇表，给特殊符号分配索引
    vocab = {token: idx for idx, token in enumerate(specials)}
    for idx, (token, _) in enumerate(counter.items(), start=len(specials)):
        vocab[token] = idx

    return vocab


# 从文件中加载句子
with open('english_sentences.txt', 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f]

with open('chinese_sentences.txt', 'r', encoding='utf-8') as f:
    chinese_sentences = [line.strip() for line in f]

# 构建词汇表
en_vocab = build_vocab(english_sentences, tokenizer_en)
zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

print(f'英文词汇表大小：{len(en_vocab)}')
print(f'中文词汇表大小：{len(zh_vocab)}')

# 将词汇表保存为 CSV 文件
pd.DataFrame(list(en_vocab.items()), columns=['Token', 'Index']).to_csv('english_vocab.csv', index=False)
pd.DataFrame(list(zh_vocab.items()), columns=['Token', 'Index']).to_csv('chinese_vocab.csv', index=False)

print("词汇表已保存为 'english_vocab.csv' 和 'chinese_vocab.csv'")


# 将句子转换为索引序列，并添加 <bos> 和 <eos>
def process_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return indices


# 处理所有句子
en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]


# 创建数据集类
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences, src_texts, trg_texts):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences
        self.src_texts = src_texts
        self.trg_texts = trg_texts

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return self.src_sequences[idx], self.trg_sequences[idx], self.src_texts[idx], self.trg_texts[idx]


# 划分训练集、验证集和测试集（80% 训练，10% 验证，10% 测试）
dataset = TranslationDataset(en_sequences, zh_sequences, english_sentences, chinese_sentences)
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f'训练集大小：{len(train_data)}')
print(f'验证集大小：{len(val_data)}')
print(f'测试集大小：{len(test_data)}')


# 保存数据集为 CSV 文件（包括原文）
def save_dataset_to_csv(data, seq_filename, text_filename):
    src_seq_data = [' '.join(map(str, src)) for src, _, _, _ in data]
    trg_seq_data = [' '.join(map(str, trg)) for _, trg, _, _ in data]
    src_text_data = [src_text for _, _, src_text, _ in data]
    trg_text_data = [trg_text for _, _, _, trg_text in data]

    # 保存索引序列
    seq_df = pd.DataFrame({'source': src_seq_data, 'target': trg_seq_data})
    seq_df.to_csv(seq_filename, index=False)
    print(f"数据集已保存为 '{seq_filename}'")

    # 保存原文句子
    text_df = pd.DataFrame({'source': src_text_data, 'target': trg_text_data})
    text_df.to_csv(text_filename, index=False)
    print(f"原文数据集已保存为 '{text_filename}'")


save_dataset_to_csv(train_data, 'train.csv', 'train_text.csv')
save_dataset_to_csv(val_data, 'val.csv', 'val_text.csv')
save_dataset_to_csv(test_data, 'test.csv', 'test_text.csv')
