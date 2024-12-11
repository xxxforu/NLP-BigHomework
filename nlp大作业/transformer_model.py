import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 加载词汇表和数据
def load_vocab(vocab_src_path, vocab_tgt_path):
    with open(vocab_src_path, 'r', encoding='utf-8') as f:
        vocab_src = {line.strip(): idx for idx, line in enumerate(f.readlines())}
    with open(vocab_tgt_path, 'r', encoding='utf-8') as f:
        vocab_tgt = {line.strip(): idx for idx, line in enumerate(f.readlines())}
    return vocab_src, vocab_tgt


# 加载数据
def load_data_from_csv(file_path):
    # 读取以tab分隔的CSV文件
    df = pd.read_csv(file_path, sep='\t', header=None)  # `sep='\t'`表示以tab作为分隔符

    # 假设每行有两个部分，第一部分是源语言，第二部分是目标语言
    src_data = df[0].apply(lambda x: str(x))  # 第0列是源语言
    tgt_data = df[1].apply(lambda x: str(x))  # 第1列是目标语言

    # 将字符串按逗号拆分成数字列表
    train_src_data = src_data.apply(lambda x: list(map(int, x.strip('[]').split(', ')))).tolist()
    train_tgt_data = tgt_data.apply(lambda x: list(map(int, x.strip('[]').split(', ')))).tolist()

    return train_src_data, train_tgt_data


# 加载训练集、验证集和测试集
vocab_src_train, vocab_tgt_train = load_vocab('vocab_src_train.txt', 'vocab_tgt_train.txt')
vocab_src_val, vocab_tgt_val = load_vocab('vocab_src_val.txt', 'vocab_tgt_val.txt')
vocab_src_test, vocab_tgt_test = load_vocab('vocab_src_test.txt', 'vocab_tgt_test.txt')

train_src_data, train_tgt_data = load_data_from_csv('train.csv')
val_src_data, val_tgt_data = load_data_from_csv('val.csv')
test_src_data, test_tgt_data = load_data_from_csv('test.csv')

# 确保 vocab_src 和 vocab_tgt 包含了所有的特殊标记
for vocab in [vocab_src_train, vocab_tgt_train, vocab_src_val, vocab_tgt_val, vocab_src_test, vocab_tgt_test]:
    vocab['<unk>'] = len(vocab)
    vocab['<pad>'] = len(vocab) + 1
    vocab['<sos>'] = len(vocab) + 2
    vocab['<eos>'] = len(vocab) + 3


# 定义 Transformer 中的 Encoder 部分
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=512):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 5000, d_model))  # 5000是最大序列长度
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

    def forward(self, src):
        embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        return self.transformer_encoder(embedded)


# 定义 Transformer 中的 Decoder 部分
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=512):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 50, d_model))  # 5000是最大序列长度
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        return self.output_layer(self.transformer_decoder(embedded, memory))


# 定义整个模型（Encoder + Decoder）
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward=512):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, nhead, num_layers, dim_feedforward)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])


# 训练过程
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Shift tgt to the right (for teacher forcing)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        # 计算损失
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# 验证过程
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)


# 训练循环
def train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# 设置超参数
d_model = 512  # Transformer的嵌入维度
nhead = 8  # 注意力头数
num_layers = 6  # 编码器和解码器的层数
dim_feedforward = 2048  # 前馈神经网络的维度

# 词汇表大小
src_vocab_size = len(vocab_src_train)
tgt_vocab_size = len(vocab_tgt_train)

# 将模型和数据加载到设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_src_train['<pad>'])

# 加载训练和验证数据
train_data = TranslationDataset(train_src_data, train_tgt_data)
val_data = TranslationDataset(val_src_data, val_tgt_data)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# 训练模型
train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10)
