import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope


# Encoder 类：手动配置参数
class Encoder(Model):
    def __init__(self, vocab_size=10000, embedding_dim=128, enc_units=256):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = Bidirectional(LSTM(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, backward_h, forward_c, backward_c = self.lstm(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        return output, state_h, state_c

    def get_config(self):
        return {
            'vocab_size': self.embedding.input_dim,
            'embedding_dim': self.embedding.output_dim,
            'enc_units': self.lstm.units
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config.get('vocab_size', 11803),
            embedding_dim=config.get('embedding_dim', 128),
            enc_units=config.get('enc_units', 256)
        )


# Decoder 类：手动配置参数
class Decoder(Model):
    def __init__(self, vocab_size=10000, embedding_dim=128, dec_units=256):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units * 2, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size, activation='softmax')

    def call(self, x, enc_output, state):
        x = self.embedding(x)
        output, h, c = self.lstm(x, initial_state=state)
        x = self.fc(output)
        return x, h, c

    def get_config(self):
        return {
            'vocab_size': self.embedding.input_dim,
            'embedding_dim': self.embedding.output_dim,
            'dec_units': self.lstm.units
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config.get('vocab_size', 7951),
            embedding_dim=config.get('embedding_dim', 128),
            dec_units=config.get('dec_units', 256)
        )


# 加载模型并注册自定义层
model_path = 'lstm_translation_model.h5'  # 修改为使用 LSTM 模型的文件

# 使用 custom_object_scope 手动加载模型并配置 Encoder 和 Decoder
with custom_object_scope({'Encoder': Encoder, 'Decoder': Decoder}):
    model = tf.keras.models.load_model(model_path)

# 加载词汇表
chinese_vocab = pd.read_csv('chinese_vocab.csv')
english_vocab = pd.read_csv('english_vocab.csv')

# 创建 Token 到 Index 的映射
chinese_token_to_idx = dict(zip(chinese_vocab['Token'], chinese_vocab['Index']))
english_token_to_idx = dict(zip(english_vocab['Token'], english_vocab['Index']))

# 反向映射 Index 到 Token
chinese_idx_to_token = {idx: token for token, idx in chinese_token_to_idx.items()}
english_idx_to_token = {idx: token for token, idx in english_token_to_idx.items()}

# 获取中文和英文词汇大小
chinese_vocab_size = len(chinese_vocab)
english_vocab_size = len(english_vocab)

# 设置模型的最大序列长度（假设已知）
max_encoder_seq_length = 50  # 需要根据你的数据调整
max_decoder_seq_length = 50  # 需要根据你的数据调整


# 对输入进行预处理
def preprocess_input(input_text, vocab, max_len):
    # 将文本拆分为单词并转换为相应的 token 索引
    sequence = [vocab.get(token, 0) for token in input_text.split()]

    # 打印出输入文本和其对应的 token 序列
    print(f"Input text: {input_text}")
    print(f"Token sequence: {sequence}")

    # 填充序列以匹配最大长度
    return pad_sequences([sequence], maxlen=max_len, padding='post')


# 设置固定的随机种子以确保一致性
np.random.seed(42)
tf.random.set_seed(42)

# 采样函数：根据温度选择下一个 token
def sample_next_token(logits, temperature=1.0):
    logits = logits / temperature  # 温度调节
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1)  # 归一化
    next_token_idx = np.random.choice(len(probs), p=probs)
    return next_token_idx


# 翻译英文到中文
def translate_english_to_chinese(input_text, temperature=0.0):
    input_sequence = preprocess_input(input_text, english_token_to_idx, max_encoder_seq_length)
    encoder_input = tf.convert_to_tensor(input_sequence, dtype=tf.int32)

    # 传递到编码器并获得编码器输出和状态
    encoder_output, state_h, state_c = model.layers[2](encoder_input)

    # 使用 <bos> token 作为解码器的初始输入
    bos_token_idx = english_token_to_idx.get('<bos>', 0)
    decoder_input = np.array([[bos_token_idx]])  # 初始输入是 <bos>

    decoded_sentence = []

    # 根据输入的 token 数量确定最大解码 token 数量
    max_decoder_len = len(input_text.split())  # 输出的 token 数量与输入的 token 数量一致

    for _ in range(max_decoder_len):
        # 获取解码器输出和新的状态
        decoder_output, decoder_state_h, decoder_state_c = model.layers[3](decoder_input, encoder_output,
                                                                           [state_h, state_c])

        # 获取预测 token
        predicted_token_idx = sample_next_token(decoder_output[0, -1, :], temperature)

        # 输出调试信息
        print(f"Predicted token index: {predicted_token_idx}, Token: {chinese_idx_to_token.get(predicted_token_idx, '<unk>')}")

        if predicted_token_idx == chinese_token_to_idx.get('<eos>', -1):
            break

        # 添加预测的 token 到解码句子中
        decoded_sentence.append(chinese_idx_to_token.get(predicted_token_idx, ''))

        # 更新解码器输入为当前预测的 token
        decoder_input = np.array([[predicted_token_idx]])

    return ' '.join(decoded_sentence)


# 翻译中文到英文
def translate_chinese_to_english(input_text, temperature=0.0):
    input_sequence = preprocess_input(input_text, chinese_token_to_idx, max_encoder_seq_length)
    encoder_input = tf.convert_to_tensor(input_sequence, dtype=tf.int32)

    encoder_output, state_h, state_c = model.layers[2](encoder_input)

    # 使用 <bos> token 作为解码器的初始输入
    bos_token_idx = chinese_token_to_idx.get('<bos>', 0)
    decoder_input = np.array([[bos_token_idx]])  # 初始输入是 <bos>

    decoded_sentence = []

    # 根据输入的 token 数量确定最大解码 token 数量
    max_decoder_len = len(input_text.split())  # 输出的 token 数量与输入的 token 数量一致

    for _ in range(max_decoder_len):
        decoder_output, decoder_state_h, decoder_state_c = model.layers[3](decoder_input, encoder_output,
                                                                           [state_h, state_c])

        predicted_token_idx = sample_next_token(decoder_output[0, -1, :], temperature)

        # 输出调试信息
        print(f"Predicted token index: {predicted_token_idx}, Token: {english_idx_to_token.get(predicted_token_idx, '<unk>')}")

        if predicted_token_idx == english_token_to_idx.get('<eos>', -1):
            break

        decoded_sentence.append(english_idx_to_token.get(predicted_token_idx, ''))

        decoder_input = np.array([[predicted_token_idx]])

    return ' '.join(decoded_sentence)


# 示例：翻译英文到中文
input_text_english = "Happy birthday to you Happy birthday to you Happy birthday to you."
translated_text_chinese = translate_english_to_chinese(input_text_english, temperature=0.7)
print(f"Input (English): {input_text_english}")
print(f"Translated (Chinese): {translated_text_chinese}")

# 示例：翻译中文到英文
input_text_chinese = "你好"
translated_text_english = translate_chinese_to_english(input_text_chinese, temperature=0.7)
print(f"Input (Chinese): {input_text_chinese}")
print(f"Translated (English): {translated_text_english}")
