import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load vocabularies
chinese_vocab = pd.read_csv('../pre_processing3/chinese_vocab.csv')
english_vocab = pd.read_csv('../pre_processing3/english_vocab.csv')

chinese_token_to_idx = dict(zip(chinese_vocab['Token'], chinese_vocab['Index']))
english_token_to_idx = dict(zip(english_vocab['Token'], english_vocab['Index']))

chinese_vocab_size = len(chinese_vocab)
english_vocab_size = len(english_vocab)


def load_data(file_path, vocab_size):
    data = pd.read_csv(file_path)
    source = [list(map(int, seq.split())) for seq in data['source']]
    target = [list(map(int, seq.split())) for seq in data['target']]

    # Filter invalid tokens
    target = [[token if token < vocab_size else 0 for token in seq] for seq in target]
    return source, target

train_source, train_target = load_data('../pre_processing3/train.csv', english_vocab_size)
val_source, val_target = load_data('../pre_processing3/val.csv', english_vocab_size)
test_source, test_target = load_data('../pre_processing3/test.csv', english_vocab_size)


# Pad sequences
max_encoder_seq_length = max(len(seq) for seq in train_source)
max_decoder_seq_length = max(len(seq) for seq in train_target)

train_source = pad_sequences(train_source, maxlen=max_encoder_seq_length, padding='post')
train_target = pad_sequences(train_target, maxlen=max_decoder_seq_length, padding='post')
val_source = pad_sequences(val_source, maxlen=max_encoder_seq_length, padding='post')
val_target = pad_sequences(val_target, maxlen=max_decoder_seq_length, padding='post')
test_source = pad_sequences(test_source, maxlen=max_encoder_seq_length, padding='post')
test_target = pad_sequences(test_target, maxlen=max_decoder_seq_length, padding='post')


# Prepare target for training
def process_target(target_sequences, max_length):
    target_input = [seq[:-1] for seq in target_sequences]
    target_output = [seq[1:] for seq in target_sequences]

    # Pad sequences to match the maximum length
    target_input = pad_sequences(target_input, maxlen=max_length, padding='post')
    target_output = pad_sequences(target_output, maxlen=max_length, padding='post')
    return np.array(target_input), np.array(target_output)


train_target_input, train_target_output = process_target(train_target, max_decoder_seq_length)
val_target_input, val_target_output = process_target(val_target, max_decoder_seq_length)

# Model definition
embedding_dim = 128
units = 256

# Encoder
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = Bidirectional(LSTM(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.lstm(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        return output, state_h, state_c

# Decoder
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units * 2, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size, activation='softmax')

    def call(self, x, enc_output, state):
        x = self.embedding(x)
        output, h, c = self.lstm(x, initial_state=state)
        x = self.fc(output)
        return x, h, c

encoder = Encoder(chinese_vocab_size, embedding_dim, units)
decoder = Decoder(english_vocab_size, embedding_dim, units)

# Model training loop
encoder_input = tf.keras.Input(shape=(max_encoder_seq_length,))
decoder_input = tf.keras.Input(shape=(max_decoder_seq_length,))

enc_output, enc_h, enc_c = encoder(encoder_input)
dec_output, _, _ = decoder(decoder_input, enc_output, [enc_h, enc_c])

model = Model([encoder_input, decoder_input], dec_output)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Train the model
history = model.fit(
    [train_source, train_target_input],
    train_target_output,
    validation_data=([val_source, val_target_input], val_target_output),
    batch_size=32,
    epochs=10
)

# Save the model
model.save('lstm_translation_model.h5')

print("Model training complete.")
