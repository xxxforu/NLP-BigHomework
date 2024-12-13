from tensorflow.keras.models import load_model

# 加载模型
model = load_model('/mnt/data/lstm_translation_model.h5', custom_objects={
    'Encoder': Encoder,
    'Decoder': Decoder
})

# 打印模型架构
model.summary()
