"""其他语言翻译至英语"""
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Project : translation
# @file    : app.py
# @Author  : yangcheng
# @Time    : 2022/8/2 15:02

import re
import json
import time
import requests
import jieba
import string
import torch
from collections import Iterable
from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# 先赋个初始值
loadModel = False
model_de2en = None  # 德语至英语
tokenizer_de2en = None
model_es2en = None  # 西班牙语至英语
tokenizer_es2en = None
model_fr2en = None  # 法语至英语
tokenizer_fr2en = None
model_it2en = None  # 意大利语至英语
tokenizer_it2en = None
model_ru2en = None  # 俄语至英语
tokenizer_ru2en = None
model_en2zh = None  # 英语至汉语
tokenizer_en2zh = None
model_zh2en = None  # 汉语至英语
tokenizer_zh2en = None
model_en2de = None  # 英语至德语
tokenizer_en2de = None
model_en2es = None  # 英语至西班牙语
tokenizer_en2es = None
model_en2fr = None  # 英语至法语
tokenizer_en2fr = None
model_en2it = None  # 英语至意大利语
tokenizer_en2it = None
model_en2ru = None  # 英语至俄语
tokenizer_en2ru = None

# 该字典建立起了百度语种判断的结果与模型语种名之间的关系
lang_map = {
    "en": "en_XX",
    "fra": "fr_XX",
    "de": "de_DE",
    "spa": "es_XX",
    "it": "it_IT",
    "zh": "zh_CN",
    "ara": "ar_AR",
    "cs": "cs_CZ",
    "est": "et_EE",
    "fin": "fi_FI",
    "hi": "hi_IN",
    "jp": "ja_XX",
    "kor": "ko_KR",
    "nl": "nl_XX",
    "ru": "ru_RU",
    "th": "th_TH",
    "ukr": "uk_UA"
}


def en_zh_inference(tokenizer_en2zh, model_en2zh, text):
    model = "./static/en_zh"  # en_zh

    tokenizer = tokenizer_en2zh

    model = model_en2zh
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    res_out = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

    indistinguishable_data = open('./data.json', 'r', encoding='utf8')
    indistinguishable_map = json.load(indistinguishable_data)["info"]
    text_split = text.split(' ')
    if len(text_split) >= 3:
        res = res_out
    else:
        if text in indistinguishable_map:
            return indistinguishable_map[text]
        else:
            intermediate = res_out
            res_list = jieba.cut(intermediate, cut_all=False)
            # 取迭代器的第一个值
            for iter in res_list:
                res = iter
                break

    return res


def filter_emoji(desstr, restr=''):
    # 过滤emoji表情
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def en2zh_inference(tokenizer_en2zh, model_en2zh, en_text):
    # t1 = time.time()
    # 读取标点符号
    punctuation = open('./punctuation.json', 'r', encoding='utf8')
    punctuation_map = json.load(punctuation)["punc"]
    # print(str(time.time()-t1)+'s')

    # 多字符切分，换行符切分相当于读取多行
    en_text = re.split('\u2060|\n', en_text)
    print(en_text)
    en_list = []
    for en in en_text:
        # 剔除首尾空字符
        en = en.strip()
        # 过滤掉''
        if en != '':
            # 判断结尾字符是否为结束标点符号
            if en[-1] not in punctuation_map:
                # 以结束标点为终止符，就在结尾添加一个'.'，然后添加到list中去
                en_list.append(en + '.')
            else:
                # 否则就直接添加到list中去
                en_list.append(en)
    # print(str(time.time()-t1)+'s')
    # print(en_list)
    # 把这些句子用换行符连接成一整段
    en = '\n'.join(en_list)
    # 过滤掉emoji表情
    en = filter_emoji(en)
    # print(str(time.time()-t1)+'s')
    # print(en)

    # 推理该句子
    # t1 = time.time()
    zh = en_zh_inference(tokenizer_en2zh, model_en2zh, en)
    # t = time.time() - t1
    # print("翻译结果为：\n{}\n耗时{}s".format(zh,t))
    return zh


def get_raw_lang(text):
    """判断语种"""
    url = "https://fanyi.baidu.com/langdetect"
    # get_request是一个字典，获取到传入的text
    get_request = {
        "query": text
    }
    # requests.get()就是来获取到指定网页的，返回一个responsed对象，这里返回的response对象携带了识别出的语种信息
    response = requests.get(url, get_request)

    # response.status_code： HTTP请求的返回状态，200表示连接成功，404表示失败
    # response.text： HTTP响应内容的字符串形式，即，url对应的页面内容
    # response.encoding：从HTTP header中猜测的响应内容编码方式
    # response.apparent_encoding：从内容中分析出的响应内容编码方式（备选编码方式）
    # response.content： HTTP响应内容的二进制形式

    # json.load()是用来读取文件的，即，将文件打开然后就可以直接读取；
    # json.loads()是用来读取字符串的，即，可以把文件打开，返回的是一个python对象，json文件对应返回的就是字典对象，这里的第一行里应该就包含了语种信息
    rt = response.text
    response_dict = json.loads(rt)

    # response_dict取lan字段对应的值raw_lang，就是识别出来的语种了，response_dict中应该有{"lan":"en"}形式的信息
    raw_lang = response_dict["lan"]
    # lang_map中存储着各种翻译逻辑，一般就是将识别出的语种翻译成另一个语言的逻辑，将识别出的语种作为键，取出对应翻译出的语种即可
    # 如果是"en"，那么就有raw_lang = "en_XX"
    raw_lang = lang_map[raw_lang]

    return raw_lang


# 推理函数inference，除了要翻译成的目标语种target和要翻译的语句text之外，还传入了一堆模型作为参数，模型开始都是None
def inference(target, text, model_de2en, tokenizer_de2en, model_es2en, tokenizer_es2en, tokenizer_en2zh, model_en2zh,
              model_fr2en, tokenizer_fr2en, model_it2en, tokenizer_it2en, model_ru2en, tokenizer_ru2en,
              model_en2de, tokenizer_zh2en, model_zh2en, tokenizer_en2de, model_en2es, tokenizer_en2es,
              model_en2fr, tokenizer_en2fr, model_en2it, tokenizer_en2it, model_en2ru, tokenizer_en2ru):
    # def inference(target, text, model_de2en, tokenizer_de2en, model_es2en,tokenizer_es2en,
    #               model_fr2en, tokenizer_fr2en, model_it2en, tokenizer_it2en, model_ru2en, tokenizer_ru2en
    #               ):
    """ translate source to english """

    # 梯度清空
    with torch.no_grad():

        # 比如判断语种"en"，返回出的就是"en_XX"
        raw_language = get_raw_lang(text)
        print("源语言为：\n", raw_language)

        # 若是中文翻译至英语
        if raw_language == "zh_CN":
            if target == "en_XX":  # 中文至英语
                print("中文至英语")
                translated = model_zh2en.generate(**tokenizer_zh2en(text, return_tensors="pt", padding=True))
                res = [tokenizer_zh2en.decode(t, skip_special_tokens=True) for t in translated][0]
                return res, raw_language

        # 1.其他语言至英语，英语再翻译至中文
        # 如果raw_language不是en_XX，指的是传入的text语种不是英语
        if raw_language != "en_XX":
            # 若raw_language是de_DE，指的是当前传入语言的语种是德语
            if raw_language == "de_DE":  # 那就希望先从德语翻译至英语
                print("德语至英语")
                # 调用模型model_de2en，tokenizer_de2en，传入text语句，返回translated
                translated = model_de2en.generate(**tokenizer_de2en(text, return_tensors="pt", padding=True))
                # 然后利用tokenizer_de2en来decode，返回的res中就包含了翻译出的英语（字符串格式）
                res = [tokenizer_de2en.decode(t, skip_special_tokens=True) for t in translated][0]
                # 继续判断，若要翻译成的语种是简体中午zh_CN
                if target == "zh_CN":
                    print("德语至中文")
                    # # 调用模型model_en2zh，tokenizer_en2zh，传入res语句，返回translated
                    # translated = model_en2zh.generate(**tokenizer_en2zh(res, return_tensors="pt", padding=True))
                    # # 继续利用tokenizer_en2zh来decode，返回的res中就包含了翻译出的中文（字符串格式）
                    # res = [tokenizer_en2zh.decode(t, skip_special_tokens=True) for t in translated][0]

                    res = en2zh_inference(tokenizer_en2zh, model_en2zh, res)

                    # 返回翻译结果res和源语言raw_language
                    return res, raw_language
                return res, raw_language
            # 西班牙语与德语一样
            elif raw_language == "es_XX":  # 西班牙语至英语、中文
                print("西班牙语至英语")
                translated = model_es2en.generate(**tokenizer_es2en(text, return_tensors="pt", padding=True))
                res = [tokenizer_es2en.decode(t, skip_special_tokens=True) for t in translated][0]
                if target == "zh_CN":
                    print("西班牙语至中文")
                    # translated = model_en2zh.generate(**tokenizer_en2zh(res, return_tensors="pt", padding=True))
                    # res = [tokenizer_en2zh.decode(t, skip_special_tokens=True) for t in translated][0]
                    res = en2zh_inference(tokenizer_en2zh, model_en2zh, res)

                    return res, raw_language
                return res, raw_language
            # 法语与德语一样
            elif raw_language == "fr_XX":  # 法语至英语、中文
                print("法语至英语")
                translated = model_fr2en.generate(**tokenizer_fr2en(text, return_tensors="pt", padding=True))
                res = [tokenizer_fr2en.decode(t, skip_special_tokens=True) for t in translated][0]
                if target == "zh_CN":
                    print("法语至中文")
                    # translated = model_en2zh.generate(**tokenizer_en2zh(res, return_tensors="pt", padding=True))
                    # res = [tokenizer_en2zh.decode(t, skip_special_tokens=True) for t in translated][0]
                    res = en2zh_inference(tokenizer_en2zh, model_en2zh, res)

                    return res, raw_language
                return res, raw_language
            # 意大利语与德语一样
            elif raw_language == "it_IT":  # 意大利语至英语、中文
                print("意大利语至英语")
                translated = model_it2en.generate(**tokenizer_it2en(text, return_tensors="pt", padding=True))
                res = [tokenizer_it2en.decode(t, skip_special_tokens=True) for t in translated][0]
                if target == "zh_CN":
                    print("意大利语至中文")
                    # translated = model_en2zh.generate(**tokenizer_en2zh(res, return_tensors="pt", padding=True))
                    # res = [tokenizer_en2zh.decode(t, skip_special_tokens=True) for t in translated][0]
                    res = en2zh_inference(tokenizer_en2zh, model_en2zh, res)

                    return res, raw_language
                return res, raw_language
            # 俄语同上
            elif raw_language == "ru_RU":  # 俄语至英语、中文
                print("俄语至英语")
                translated = model_ru2en.generate(**tokenizer_ru2en(text, return_tensors="pt", padding=True))
                res = [tokenizer_ru2en.decode(t, skip_special_tokens=True) for t in translated][0]
                if target == "zh_CN":
                    print("俄语至中文")
                    # translated = model_en2zh.generate(**tokenizer_en2zh(res, return_tensors="pt", padding=True))
                    # res = [tokenizer_en2zh.decode(t, skip_special_tokens=True) for t in translated][0]
                    res = en2zh_inference(tokenizer_en2zh, model_en2zh, res)

                    return res, raw_language
                return res, raw_language
            # 不支持其他语言的翻译
            else:
                res = "不支持该语言的翻译"
                return res, raw_language

        # 2.源语言为英语，翻译从英语至其他语种
        elif raw_language == "en_XX":
            if target == "de_DE":  # 英语至德语
                print("英语至德语")
                translated = model_en2de.generate(**tokenizer_en2de(text, return_tensors="pt", padding=True))
                res = [tokenizer_en2de.decode(t, skip_special_tokens=True) for t in translated][0]
                return res, raw_language

            elif target == "es_XX":  # 英语至西班牙语
                print("英语至西班牙语")
                translated = model_en2es.generate(**tokenizer_en2es(text, return_tensors="pt", padding=True))
                res = [tokenizer_en2es.decode(t, skip_special_tokens=True) for t in translated][0]
                return res, raw_language

            elif target == "fr_XX":  # 英语至法语
                print("英语至法语")
                translated = model_en2fr.generate(**tokenizer_en2fr(text, return_tensors="pt", padding=True))
                res = [tokenizer_en2fr.decode(t, skip_special_tokens=True) for t in translated][0]
                return res, raw_language

            elif target == "it_IT":  # 英语至意大利语
                print("英语至意大利语")
                translated = model_en2it.generate(**tokenizer_en2it(text, return_tensors="pt", padding=True))
                res = [tokenizer_en2it.decode(t, skip_special_tokens=True) for t in translated][0]
                return res, raw_language

            elif target == "ru_RU":  # 英语至俄语
                print("英语至俄语")
                translated = model_en2ru.generate(**tokenizer_en2ru(text, return_tensors="pt", padding=True))
                res = [tokenizer_en2ru.decode(t, skip_special_tokens=True) for t in translated][0]
                return res, raw_language

            elif target == "zh_CN":  # 英语至中文
                print("英语至中文")
                res = en2zh_inference(tokenizer_en2zh, model_en2zh, text)

                return res, raw_language
            # print("英语至中文")
            # res = en_zh_inference(text)

        else:
            res = "不支持该语言至目标语言的翻译"
            return res, raw_language


@app.route('/')
def hello_world():  # put application's code here
    return '文本翻译项目'


@app.route("/v1/predict", methods=["POST", "GET"])
def predict():
    """Provide main prediction API route. """
    t1 = time.time()
    try:
        global loadModel

        global model_de2en  # 德语至英语
        global tokenizer_de2en

        global model_es2en  # 西班牙语至英语
        global tokenizer_es2en

        global model_fr2en  # 法语至英语
        global tokenizer_fr2en

        global model_it2en  # 意大利语至英语
        global tokenizer_it2en

        global model_ru2en  # 俄语至英语
        global tokenizer_ru2en

        global model_en2zh  # 英语至汉语
        global tokenizer_en2zh

        global model_zh2en  # 汉语至英语
        global tokenizer_zh2en

        global model_en2de  # 英语至德语
        global tokenizer_en2de

        global model_en2es  # 英语至西班牙语
        global tokenizer_en2es

        global model_en2fr  # 英语至法语
        global tokenizer_en2fr

        global model_en2it  # 英语至意大利语
        global tokenizer_en2it

        global model_en2ru  # 英语至俄语
        global tokenizer_en2ru

        # 用户输入的数据以http协议中的post方式发送请求过来，request对象中封装了浏览器端发送过来的请求信息
        # request.get_json()可以获取到以表头为application/json格式发送过来的请求表单，用json.dumps()将其解析为json字符串格式
        # data = json.dumps(request.get_json())  # str

        data = request.get_data(as_text=True)
        # json.loads()加载这个json字符串，返回的是字典对象
        data = json.loads(data)

        # 查看data的"tar"键中有没有值，若判空
        if not data["tar"]:
            # 返回异常信息，其实jsonify与json.dumps()作用类似，但是jsonify会返回一个请求头为application/json的json数据，而json.dumps()返回的是text/html的请求头数据
            return jsonify({
                "code": 400,
                "message": "目标语种为空"
            })

        # 文本判空
        if not data["text"]:
            return jsonify({
                "code": 400,
                "message": "文本为空"
            })

        # 若loadModel为False，就执行
        if not loadModel:
            print("加载一次")
            # 德语至英语
            # 加载对应的预训练模型
            tokenizer_de2en = MarianTokenizer.from_pretrained("./static/de_en")
            model_de2en = MarianMTModel.from_pretrained("./static/de_en")
            # 让模型进入eval模式，注意train模式时，会进行BN与DropOut操作，这是为了训练，而eval模式是为了推理，所有会关闭BN与DropOut操作，而直接采用已保存模型中的参数值
            model_de2en.eval()

            # 以下语言同上
            # 西班牙语至英语
            tokenizer_es2en = MarianTokenizer.from_pretrained("./static/es_en")
            model_es2en = MarianMTModel.from_pretrained("./static/es_en")
            model_es2en.eval()

            # 法语至英语
            tokenizer_fr2en = MarianTokenizer.from_pretrained("./static/fr_en")
            model_fr2en = MarianMTModel.from_pretrained("./static/fr_en")
            model_fr2en.eval()

            # 意大利语至英语
            tokenizer_it2en = MarianTokenizer.from_pretrained("./static/it_en")
            model_it2en = MarianMTModel.from_pretrained("./static/it_en")
            model_it2en.eval()

            # 俄语至英语
            tokenizer_ru2en = MarianTokenizer.from_pretrained("./static/ru_en")
            model_ru2en = MarianMTModel.from_pretrained("./static/ru_en")
            model_ru2en.eval()

            # 英语至汉语
            tokenizer_en2zh = MarianTokenizer.from_pretrained("./static/en_zh")
            model_en2zh = MarianMTModel.from_pretrained("./static/en_zh")
            model_en2zh.eval()

            # 汉语至英语
            tokenizer_zh2en = MarianTokenizer.from_pretrained("./static/zh_en")
            model_zh2en = MarianMTModel.from_pretrained("./static/zh_en")
            model_zh2en.eval()

            # 英语至德语
            tokenizer_en2de = MarianTokenizer.from_pretrained("./static/en_de")
            model_en2de = MarianMTModel.from_pretrained("./static/en_de")
            model_en2de.eval()

            # 英语至西班牙语
            tokenizer_en2es = MarianTokenizer.from_pretrained("./static/en_es")
            model_en2es = MarianMTModel.from_pretrained("./static/en_es")
            model_en2es.eval()

            # 英语至法语
            tokenizer_en2fr = MarianTokenizer.from_pretrained("./static/en_fr")
            model_en2fr = MarianMTModel.from_pretrained("./static/en_fr")
            model_en2fr.eval()

            # 英语至意大利语
            tokenizer_en2it = MarianTokenizer.from_pretrained("./static/en_it")
            model_en2it = MarianMTModel.from_pretrained("./static/en_it")
            model_en2it.eval()

            # 英语至俄语
            tokenizer_en2ru = MarianTokenizer.from_pretrained("./static/en_ru")
            model_en2ru = MarianMTModel.from_pretrained("./static/en_ru")
            model_en2ru.eval()

        # 加载完模型，开始推理，传入指定参数即可
        res, raw_language = inference(data["tar"], data["text"], model_de2en, tokenizer_de2en, model_es2en,
                                      tokenizer_es2en, tokenizer_en2zh, model_en2zh,
                                      model_fr2en, tokenizer_fr2en, model_it2en, tokenizer_it2en, model_ru2en,
                                      tokenizer_ru2en,
                                      model_en2de, tokenizer_zh2en, model_zh2en, tokenizer_en2de, model_en2es,
                                      tokenizer_en2es,
                                      model_en2fr, tokenizer_en2fr, model_en2it, tokenizer_en2it, model_en2ru,
                                      tokenizer_en2ru)

        # 成功以后不再加载（是不是@app.run下的程序会一遍遍地按顺序执行，所以要在这里赋值为True，以保证执行完一次之后，就进不来这个判断了，否则会不断重复进行翻译）
        loadModel = True

        # 返回内容
        return_dict = {
            'code': 200,
            'raw_language': raw_language,
            'language': data["tar"],
            'raw_text': str(data["text"]),
            'result': str(res),
        }

        t = time.time() - t1
        print("执行完毕，用时{}s".format(t))

        return json.dumps(return_dict, ensure_ascii=False)
    except Exception as ex:
        return json.dumps({"error": str(ex)}, ensure_ascii=False)


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == '__main__':
    main()