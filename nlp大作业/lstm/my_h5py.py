# -*- coding: utf-8 -*-
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", torch_dtype="auto")



text = "从时间上看，中国空间站的建造比国际空间站晚20多年。"
# Tokenize the text
batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])

# Make sure that the tokenized text does not exceed the maximum
# allowed size of 512
batch["input_ids"] = batch["input_ids"][:, :512]
batch["attention_mask"] = batch["attention_mask"][:, :512]

# Perform the translation and decode the output
translation = model.generate(**batch)
result = tokenizer.batch_decode(translation, skip_special_tokens=True)
print(result)
