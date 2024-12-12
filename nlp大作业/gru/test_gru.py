import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope


# Encoder 类：手动配置参数
class Encoder(Model):
    def __init__(self, vocab_size=10000, embedding_dim=128, enc_units=256):
        super(Encoder, self).__init__()
        # 使用默认值或传入的参数来配置
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = Bidirectional(GRU(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, backward_h = self.gru(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        return output, state_h

    def get_config(self):
        # 这里手动返回配置，不依赖于 config 中的值
        return {
            'vocab_size': self.embedding.input_dim,
            'embedding_dim': self.embedding.output_dim,
            'enc_units': self.gru.units
        }

    @classmethod
    def from_config(cls, config):
        # 手动配置 Encoder，使用默认值或传入的配置
        return cls(
            vocab_size=config.get('vocab_size', 11803),
            embedding_dim=config.get('embedding_dim', 128),
            enc_units=config.get('enc_units', 256)
        )


# Decoder 类：手动配置参数
class Decoder(Model):
    def __init__(self, vocab_size=10000, embedding_dim=128, dec_units=256):
        super(Decoder, self).__init__()
        # 使用默认值或传入的参数来配置
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(dec_units * 2, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size, activation='softmax')

    def call(self, x, enc_output, state):
        x = self.embedding(x)
        output, h = self.gru(x, initial_state=state)
        x = self.fc(output)
        return x, h

    def get_config(self):
        # 这里手动返回配置，不依赖于 config 中的值
        return {
            'vocab_size': self.embedding.input_dim,
            'embedding_dim': self.embedding.output_dim,
            'dec_units': self.gru.units
        }

    @classmethod
    def from_config(cls, config):
        # 手动配置 Decoder，使用默认值或传入的配置
        return cls(
            vocab_size=config.get('vocab_size', 7951),
            embedding_dim=config.get('embedding_dim', 128),
            dec_units=config.get('dec_units', 256)
        )



# 加载模型并注册自定义层
model_path = 'gru_translation_model.h5'

# 手动创建 config 字典
encoder_config = {
    'vocab_size': 10000,  # 您的词汇大小
    'embedding_dim': 128,  # 嵌入维度
    'enc_units': 256      # 编码器 GRU 单元数
}

decoder_config = {
    'vocab_size': 10000,  # 您的词汇大小
    'embedding_dim': 128,  # 嵌入维度
    'dec_units': 256      # 解码器 GRU 单元数
}

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
max_encoder_seq_length = 100  # 需要根据你的数据调整
max_decoder_seq_length = 100  # 需要根据你的数据调整


# 对输入进行预处理
def preprocess_input(input_text, vocab, max_len):
    sequence = [vocab.get(token, 0) for token in input_text.split()]
    return pad_sequences([sequence], maxlen=max_len, padding='post')


# 翻译函数
def translate(input_text):
    # 预处理输入英文
    input_sequence = preprocess_input(input_text, english_token_to_idx, max_encoder_seq_length)

    # 解码时的初始状态（初始的编码器输出和状态）
    encoder_input = tf.convert_to_tensor(input_sequence, dtype=tf.int32)

    # 获取编码器的输出和状态
    encoder_output, encoder_state = model.layers[2](encoder_input)  # 获取 Encoder 的输出
    decoder_input = np.zeros((1, 1))  # 解码器的初始输入是一个填充的零
    decoded_sentence = []

    # 解码循环，直到生成 [END] 词或达到最大长度
    for _ in range(max_decoder_seq_length):
        # 调用解码器并生成下一个词
        decoder_output, decoder_state = model.layers[3](decoder_input, encoder_output, encoder_state)

        # 获取预测词的索引
        predicted_token_idx = np.argmax(decoder_output[0, -1, :])

        # 如果预测的词是 [END]，则停止
        if predicted_token_idx == chinese_token_to_idx.get('[END]', -1):
            break

        # 将预测的词添加到解码序列中
        decoded_sentence.append(chinese_idx_to_token.get(predicted_token_idx, ''))

        # 更新解码器输入（输入是上一步的预测词）
        decoder_input = np.array([[predicted_token_idx]])

    # 返回翻译后的中文句子
    return ' '.join(decoded_sentence)


# 示例
input_text = "<bos> Choose one person. <eos>"
translated_text = translate(input_text)
print(f"Input (English): {input_text}")
print(f"Translated (Chinese): {translated_text}")
