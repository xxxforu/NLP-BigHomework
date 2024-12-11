import random

# 1. 读取对齐后的数据
def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

# 2. 划分数据集：训练集 80%，验证集 10%，测试集 10%
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)  # 打乱数据顺序

    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data

# 3. 保存数据到文件
def save_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(data)

# 4. 主程序
def main():
    # 读取原始对齐数据
    file_path = "clean3.txt"  # 替换为你的文件路径
    data = read_data(file_path)

    # 划分数据集
    train_data, val_data, test_data = split_data(data)

    # 保存到不同文件
    save_data(train_data, "train.txt")
    save_data(val_data, "val.txt")
    save_data(test_data, "test.txt")

    print(f"Data has been split into: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples.")

if __name__ == "__main__":
    main()
