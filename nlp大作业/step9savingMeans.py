# 将编码后的数据存储为不同格式
import pandas as pd
import torch

# Step 1: 读取编码后的数据
file_path = 'clean8.txt'  # 替换为您的文件路径
df = pd.read_csv(file_path, sep='\t', header=None)

# Step 2: 将数据转换为列表
sources = df[0].tolist()  # 获取 source 列
targets = df[1].tolist()  # 获取 target 列

# Step 3: 保存为 PyTorch Tensor
torch.save((sources, targets), 'dataset.pt')
print("数据已保存为 dataset.pt")

# Step 4: 保存为 JSON
df.to_json('dataset.json', orient='records', lines=True)
print("数据已保存为 dataset.json")

# Step 5: 保存为 CSV
df.to_csv('dataset.csv', index=False)
print("数据已保存为 dataset.csv")
