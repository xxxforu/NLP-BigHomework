import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
from nltk.tokenize import word_tokenize

from lstmModel import Encoder  # 确保路径和类名正确

# 正确加载模型
try:
    model = load_model('encoder_decoder_weights.weights.h5', custom_objects={'Encoder': Encoder})
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型时发生错误: {e}")



# 加载词汇表
chinese_vocab = pd.read_csv('../pre_processing3/chinese_vocab.csv')
english_vocab = pd.read_csv('../pre_processing3/english_vocab.csv')

# 创建词汇映射
chinese_token_to_idx = dict(zip(chinese_vocab['Token'], chinese_vocab['Index']))
english_idx_to_token = dict(zip(english_vocab['Index'], english_vocab['Token']))

# 最大序列长度（需与训练时保持一致）
max_encoder_seq_length = 30  # 替换为实际的编码器最大序列长度
max_decoder_seq_length = 30  # 替换为实际的解码器最大序列长度

# 翻译函数
def translate(text, is_chinese=True):
    """
    使用训练好的模型翻译文本。
    :param text: 输入的句子（中文或英文）
    :param is_chinese: 如果是中文翻译到英文，则为 True；否则为 False。
    :return: 翻译后的句子
    """
    if is_chinese:
        input_vocab = chinese_token_to_idx
        output_vocab = english_idx_to_token
        tokenizer = jieba.cut
    else:
        input_vocab = {row['Token']: row['Index'] for _, row in english_vocab.iterrows()}
        output_vocab = {row['Index']: row['Token'] for _, row in chinese_vocab.iterrows()}
        tokenizer = word_tokenize

    # 分词与映射
    input_tokens = [input_vocab.get(token, input_vocab['<unk>']) for token in tokenizer(text)]
    input_seq = pad_sequences([input_tokens], maxlen=max_encoder_seq_length, padding='post')

    # 编码器的输出
    enc_output, enc_h, enc_c = model.layers[2](input_seq)

    # 解码器初始输入（起始标记 <bos> 的索引）
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = input_vocab['<bos>']

    stop_condition = False
    translated_tokens = []

    # 开始解码
    while not stop_condition:
        dec_output, dec_h, dec_c = model.layers[3](target_seq, enc_output, [enc_h, enc_c])
        predicted_idx = np.argmax(dec_output[0, -1, :])

        sampled_token = output_vocab.get(predicted_idx, '<unk>')
        if sampled_token == '<eos>' or len(translated_tokens) > max_decoder_seq_length:
            stop_condition = True
        else:
            translated_tokens.append(sampled_token)

        # 更新解码器的输入和状态
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = predicted_idx
        enc_h, enc_c = dec_h, dec_c

    return ' '.join(translated_tokens)

# 主循环
print("模型已加载。输入句子开始翻译，输入 'quit' 退出。")
while True:
    user_input = input("输入句子: ")
    if user_input.lower() == 'quit':
        break
    # 判断是中译英还是英译中
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in user_input)
    translated_sentence = translate(user_input, is_chinese=is_chinese)
    print(f"翻译结果: {translated_sentence}")
