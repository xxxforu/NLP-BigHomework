from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional
from tensorflow.keras.models import Model

app = Flask(__name__)

# 定义 Encoder 类
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = Bidirectional(GRU(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, backward_h = self.gru(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        return output, state_h

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'enc_units': self.enc_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 定义 Decoder 类
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(dec_units * 2, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size, activation='softmax')

    def call(self, x, enc_output, state):
        x = self.embedding(x)
        output, h = self.gru(x, initial_state=state)
        x = self.fc(output)
        return x, h

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dec_units': self.dec_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# 加载模型
lstm_model = load_model('models/lstm_translation_model.h5', custom_objects={'Encoder': Encoder, 'Decoder': Decoder})
gru_model = load_model('models/gru_translation_model.h5', custom_objects={'Encoder': Encoder, 'Decoder': Decoder})


# 加载词汇表
chinese_vocab = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in open('../pre_processing3/chinese_vocab.csv')}
english_vocab = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in open('../pre_processing3/english_vocab.csv')}

# 反向词典：将索引映射回单词
idx_to_english_token = {v: k for k, v in english_vocab.items()}
idx_to_chinese_token = {v: k for k, v in chinese_vocab.items()}

# 定义最大序列长度（与训练时保持一致）
max_encoder_seq_length = 20  # 修改为你实际的最大长度
max_decoder_seq_length = 20  # 修改为你实际的最大长度

def preprocess_input(text, vocab, max_length):
    tokens = [vocab.get(word, 0) for word in text.split()]
    return pad_sequences([tokens], maxlen=max_length, padding='post')

def translate(text, model, model_type):
    input_seq = preprocess_input(text, chinese_vocab, max_encoder_seq_length)

    # 构建解码输入（<start> token 或初始状态，假设为 0）
    decoder_input = np.zeros((1, max_decoder_seq_length))

    # 翻译过程
    enc_output, enc_state = model.layers[2](input_seq)  # Encoder 的索引视情况而定
    dec_output, _ = model.layers[3](decoder_input, enc_output, enc_state)  # Decoder 的索引视情况而定

    predicted_ids = np.argmax(dec_output[0], axis=-1)
    translated_tokens = [idx_to_english_token.get(idx, '<UNK>') for idx in predicted_ids]
    return ' '.join(translated_tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    input_text = data['text']
    model_type = data['model']  # 'lstm' 或 'gru'

    if model_type == 'lstm':
        translated_text = translate(input_text, lstm_model, model_type)
    elif model_type == 'gru':
        translated_text = translate(input_text, gru_model, model_type)
    else:
        return jsonify({'error': 'Invalid model type selected'}), 400

    return jsonify({'translation': translated_text})

if __name__ == '__main__':
    app.run(debug=True)
