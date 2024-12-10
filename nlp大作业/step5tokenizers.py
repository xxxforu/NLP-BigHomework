# 进行标记化和子词切分
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

# 文件路径
input_file = "clean4.txt"
output_file = "clean5.txt"

# 加载子词分词器 (以BERT模型为例)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 读取文件
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 处理文本
processed_lines = []
for line in lines:
    if "\t" in line:
        english_text, chinese_text = line.split("\t")

        # 英文标记化
        tokens = word_tokenize(english_text)

        # 子词切分
        subwords = tokenizer.tokenize(" ".join(tokens))

        # 重组文本
        processed_line = f"{' '.join(subwords)}\t{chinese_text.strip()}\n"
        processed_lines.append(processed_line)

# 保存结果
with open(output_file, "w", encoding="utf-8") as file:
    file.writelines(processed_lines)

print(f"处理完成，结果已保存到 {output_file}")
