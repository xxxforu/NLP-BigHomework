import re
import opencc


def preprocess_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = set()  # 用set去除重复的行
    cleaned_lines = []

    for line in lines:
        # 移除空行
        if not line.strip():
            continue

        # 提取中英文部分
        match = re.match(r'([^\u4e00-\u9fa5]+)([\u4e00-\u9fa5]+)', line.strip())
        if match:
            source_text = match.group(1).strip()  # 英文部分
            target_text = match.group(2).strip()  # 中文部分

            # 检查句子长度是否在有效范围内
            if len(source_text) < 2 or len(source_text) > 100 or len(target_text) < 2 or len(target_text) > 100:
                continue  # 跳过不符合条件的样本


            # 拼接清理后的行，并加入set去重
            cleaned_line = source_text + '\t' + target_text
            processed_lines.add(cleaned_line)

    # 将去重后的内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')

# 使用示例
input_file = 'clean2.txt'  # 输入文件名
output_file = 'clean3.txt'  # 输出文件名
preprocess_text(input_file, output_file)
