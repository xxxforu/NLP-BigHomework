import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Load vocabularies
chinese_vocab = pd.read_csv("../pre_processing3/chinese_vocab.csv")
english_vocab = pd.read_csv('../pre_processing3/english_vocab.csv')

chinese_token_to_idx = dict(zip(chinese_vocab['Token'], chinese_vocab['Index']))
english_idx_to_token = dict(zip(english_vocab['Index'], english_vocab['Token']))

chinese_vocab_size = len(chinese_vocab)
english_vocab_size = len(english_vocab)

# Load data
def load_data(file_path, vocab_size):
    data = pd.read_csv(file_path)
    source = [list(map(int, seq.split())) for seq in data['source']]
    target = [list(map(int, seq.split())) for seq in data['target']]
    target = [[token if token < vocab_size else 0 for token in seq] for seq in target]
    return source, target

train_source, train_target = load_data('../pre_processing3/train.csv', english_vocab_size)
val_source, val_target = load_data('../pre_processing3/val.csv', english_vocab_size)

def pad_sequences(sequences, maxlen):
    return [seq + [0] * (maxlen - len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in sequences]

# Pad sequences
max_encoder_seq_length = max(len(seq) for seq in train_source)
max_decoder_seq_length = max(len(seq) for seq in train_target)

train_source = pad_sequences(train_source, max_encoder_seq_length)
train_target = pad_sequences(train_target, max_decoder_seq_length)

# Prepare target for training
def process_target(target_sequences, max_length):
    target_input = [seq[:-1] for seq in target_sequences]
    target_output = [seq[1:] for seq in target_sequences]
    target_input = pad_sequences(target_input, max_length)
    target_output = pad_sequences(target_output, max_length)
    return torch.tensor(target_input, dtype=torch.long), torch.tensor(target_output, dtype=torch.long)

train_target_input, train_target_output = process_target(train_target, max_decoder_seq_length)

class TranslationDataset(Dataset):
    def __init__(self, source, target_input, target_output):
        self.source = torch.tensor(source, dtype=torch.long)
        self.target_input = target_input
        self.target_output = target_output

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target_input[idx], self.target_output[idx]

train_dataset = TranslationDataset(train_source, train_target_input, train_target_output)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 修改 Encoder forward 方法，返回正向 LSTM 状态用于解码器初始化
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (h, c) = self.lstm(x)
        # 调整双向隐状态的形状，使其适配解码器
        h = h.view(2, -1, h.size(2))
        c = c.view(2, -1, c.size(2))
        h = h.sum(dim=0, keepdim=True)
        c = c.sum(dim=0, keepdim=True)
        return outputs, (h, c)

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        hidden = hidden[-1].unsqueeze(1).repeat(1, max_len, 1)  # Expand hidden state
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention, dim=1)

# 修改 Decoder 增加注意力机制
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        x = self.embedding(x)
        attn_weights = self.attention(hidden[0], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.repeat(1, x.size(1), 1)  # Repeat context across all time steps
        lstm_input = torch.cat((x, context), dim=2)
        outputs, hidden = self.lstm(lstm_input, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden

# 初始化模型
embedding_dim = 128
hidden_size = 256

encoder = Encoder(chinese_vocab_size, embedding_dim, hidden_size)
decoder = Decoder(english_vocab_size, embedding_dim, hidden_size)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 PAD 索引
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
for epoch in range(10):
    encoder.train()
    decoder.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    print(f"Starting Epoch {epoch + 1}...")
    for batch_idx, (source, target_input, target_output) in enumerate(train_loader):
        optimizer.zero_grad()
        enc_output, hidden = encoder(source)
        dec_output, _ = decoder(target_input, hidden, enc_output)

        loss = criterion(dec_output.reshape(-1, english_vocab_size), target_output.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        predictions = torch.argmax(dec_output, dim=2)
        correct_predictions += (predictions == target_output).sum().item()
        total_predictions += target_output.numel()

        if (batch_idx + 1) % 10 == 0:
            accuracy = correct_predictions / total_predictions
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Save the models
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")

# Function to translate
def translate(text):
    tokenized_text = [chinese_token_to_idx.get(token, chinese_token_to_idx["<unk>"]) for token in text.split()]
    encoder_input_seq = torch.tensor([tokenized_text + [0] * (max_encoder_seq_length - len(tokenized_text))], dtype=torch.long)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        enc_output, hidden = encoder(encoder_input_seq)
        target_seq = torch.tensor([[chinese_token_to_idx["<start>"]]], dtype=torch.long)

        translated = []
        for _ in range(max_decoder_seq_length):
            dec_output, hidden = decoder(target_seq, hidden, enc_output)
            predicted_id = torch.argmax(dec_output.squeeze(1), dim=1).item()
            translated.append(predicted_id)

            if predicted_id == chinese_token_to_idx["<end>"]:
                break

            target_seq = torch.tensor([[predicted_id]], dtype=torch.long)

    return " ".join(english_idx_to_token.get(idx, "<unk>") for idx in translated)

# Example usage
print("Model training complete. Enter 'quit' to exit.")
while True:
    user_input = input("Enter text to translate (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    translated_text = translate(user_input)
    print("Translated Text:", translated_text)
