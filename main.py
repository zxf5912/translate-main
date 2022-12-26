#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Project : translation
# @file    : singleMain.py
# @Author  : yangcheng
# @Time    : 2022/8/15 10:01
import time

from transformers import MarianMTModel, MarianTokenizer


def inference(language, text):
    start = time.time()

    # model name
    model = "./ckpt/" + language + "_en"  # es西班牙语，fr法语，de德语，it法语

    tokenizer = MarianTokenizer.from_pretrained(model)

    model = MarianMTModel.from_pretrained(model)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    # print(res[0])
    # print(time.time() - start)

    return res[0]
