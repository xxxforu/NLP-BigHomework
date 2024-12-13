from tensorflow.keras.layers import Embedding, GRU, Bidirectional, Concatenate,LSTM
from tensorflow.keras.models import Model

class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = Bidirectional(GRU(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        output, forward_h, backward_h = self.gru(x)
        state_h = Concatenate()([forward_h, backward_h])
        return output, state_h


    class Encoder(Model):
        def __init__(self, vocab_size, embedding_dim, enc_units):
            super(Encoder, self).__init__()
            self.embedding = Embedding(vocab_size, embedding_dim)
            self.lstm = Bidirectional(LSTM(enc_units, return_sequences=True, return_state=True))

        def call(self, x):
            x = self.embedding(x)
            output, forward_h, forward_c, backward_h, backward_c = self.lstm(x)
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            return output, state_h, state_c

