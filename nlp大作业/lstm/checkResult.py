# 1.加载模型
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('lstm_translation_model.h5')

# 查看模型结构
model.summary()

# 2.查看模型架构
with open('model_structure.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

