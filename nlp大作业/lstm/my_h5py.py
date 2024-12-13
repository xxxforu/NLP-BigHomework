import my_h5py

# 打开模型文件
class File:
    pass


with h5py.File('lstm_translation_model.h5', 'r') as f:
    # 查看顶层键
    print(f.keys())
    # 查看某个键的内容
    print(f['model_weights'].keys())
