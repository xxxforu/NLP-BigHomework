import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request

from tensorflow.keras.models import load_model

from nlp大作业.gru.gru import Encoder as GRUEncoder
from nlp大作业.lstm.lstmModel import Encoder as LSTMEncoder

# 引入自定义的 GRU Encoder 类
gru_model = load_model('models/gru_translation_model.h5', custom_objects={'Encoder': GRUEncoder})

# 引入自定义的 LSTM Encoder 类
lstm_model = load_model('models/lstm_translation_model.h5', custom_objects={'Encoder': LSTMEncoder})


# 加载词汇表
chinese_vocab = pd.read_csv('../pre_processing3/chinese_vocab.csv')
english_vocab = pd.read_csv('../pre_processing3/english_vocab.csv')

chinese_token_to_idx = dict(zip(chinese_vocab['Token'], chinese_vocab['Index']))
english_token_to_idx = dict(zip(english_vocab['Token'], english_vocab['Index']))

# 逆词典，方便将预测结果转换为文本
idx_to_english_token = {v: k for k, v in english_token_to_idx.items()}

# 最大序列长度
max_decoder_seq_length = len(english_vocab)


# 文本处理函数
def translate(input_text, model):
    # 将输入文本转换为索引
    input_seq = [chinese_token_to_idx.get(token, 0) for token in input_text]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_decoder_seq_length,
                                                              padding='post')

    # 初始化解码器的输入
    target_seq = np.zeros((1, max_decoder_seq_length))
    target_seq[0, 0] = english_token_to_idx.get('<start>', 0)  # 假设'<start>'是开始符

    # 存储翻译结果
    translated_sentence = []

    # 逐步预测
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = model.predict([input_seq, target_seq])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # 如果遇到结束符，停止解码
        if sampled_token_index == english_token_to_idx.get('<eos>', 1):  # 假设'<end>'是结束符
            break

        # 将预测的索引添加到翻译结果中
        translated_sentence.append(idx_to_english_token[sampled_token_index])

        # 更新解码器的输入
        target_seq[0, len(translated_sentence)] = sampled_token_index

    return ' '.join(translated_sentence)


# 创建Flask应用
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    translation = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        model_choice = request.form['model_choice']

        if model_choice == 'GRU':
            translation = translate(input_text, gru_model)
        elif model_choice == 'LSTM':
            translation = translate(input_text, lstm_model)

    return render_template('index.html', translation=translation)


if __name__ == '__main__':
    app.run(debug=True)
