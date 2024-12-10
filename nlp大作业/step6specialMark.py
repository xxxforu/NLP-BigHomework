# 进行特殊标记：开始标记<sos>和结束标记<eos>
import pandas as pd

# Step 1: 读取文件
file_path = 'clean5.txt'  # 替换为您的文件路径
# 使用 header=None 来表示没有列名，并将其读入为 DataFrame
df = pd.read_csv(file_path, sep='\t', header=None)

# Step 2: 打印读取的数据以确认格式
print(df.head())  # 打印前几行数据，确认格式正确

# Step 3: 添加特殊标记
# 由于没有列名，我们可以直接使用 df[0] 和 df[1] 访问数据
df[0] = df[0].apply(lambda x: '<sos> ' + x + ' <eos>')
df[1] = df[1].apply(lambda x: '<sos> ' + x + ' <eos>')

# Step 4: 保存结果
output_file_path = 'clean6.txt'  # 设置输出文件名
df.to_csv(output_file_path, sep='\t', index=False, header=False)

print(f"处理完成，结果已保存到 {output_file_path}")