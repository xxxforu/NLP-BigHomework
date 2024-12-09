import re


def preprocess_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = set()  # 用set去除重复的行
    cleaned_lines = []

    for line in lines:
        # 移除空行
        if not line.strip():
            continue

        # 提取中文部分，删除中文后面的多余内容
        match = re.match(r'([^\u4e00-\u9fa5]+)([\u4e00-\u9fa5]+)', line.strip())
        if match:
            cleaned_line = match.group(1).strip() + '\t' + match.group(2).strip()
            # 将处理后的行加入set中去重
            processed_lines.add(cleaned_line)

    # 将去重后的内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')


# 使用示例
input_file = 'cmn.txt'  # 输入文件名
output_file = 'clean1.txt'  # 输出文件名
preprocess_text(input_file, output_file)
