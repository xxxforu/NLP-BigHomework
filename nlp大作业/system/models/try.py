
from tensorflow.keras.models import load_model

from nlp大作业.lstm.lstmModel import Encoder
# 引入自定义的 LSTM Encoder 类
lstm_model = load_model('lstm_translation_model.h5')

lstm_model.summary()