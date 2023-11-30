import pandas as pd
import os
import time
import jieba.analyse
import string
from itertools import permutations
import re
import torch
from ltp import LTP
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch')
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

normalized_levenshtein = NormalizedLevenshtein()

jieba.analyse.set_stop_words("stopwords")
jieba.load_userdict("userdict")

ltp = LTP("LTP/base")
# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

words = []
with open("userdict", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        words.append(line.strip())
ltp.add_words(words)

stopwords = []
with open("cn_stopwords.txt", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        stopwords.append(line.strip())


def ch_text_pro1(str1):
    # 针对 中文
    # 去除一些字符，替换多个空格为单个，字母取小写
    str2 = str1.replace("\n", " ").replace("\t", " ").replace("\xa0", "")
    str3 = re.sub(' +', '', str2).strip()
    str4 = str3.lower()
    return str4


# jieba的关键词抽取基于tfidf
def return_tfidf_kw(str1):
    result = []
    for kw, value in jieba.analyse.extract_tags(str1, withWeight=True, topK=10):
        result.append(kw)
    return result


# jieba的关键词抽取基于textrank
def return_textrank_kw(str1):
    result = []
    for kw, value in jieba.analyse.textrank(str1, withWeight=True, topK=10):
        result.append(kw)
    return result


# 关键词抽取结果拓展，组合便利
def keyword_extraction(content_str):
    content = ch_text_pro1(content_str)
    tfidf_kw = return_tfidf_kw(content)
    keyword_list = tfidf_kw
    # 对keyword_list0进行排列组合，匹配到的组合加入keyword_list
    mylist = list(permutations(tfidf_kw, 2))
    for m in mylist:
        long_str = m[0] + m[1]
        count = len(content.split(long_str))
        if count > 1:
            keyword_list.append(long_str)
        else:
            continue
    return keyword_list


# LTP的语义角色标注
def srl_extraction(content_str):
    content = ch_text_pro1(content_str)
    output = ltp.pipeline([content + "？"], tasks=["cws", "pos", "srl"])
    srl_result = output.srl
    arguments_list = {}
    for result in srl_result:
        for res in result:
            arguments = res["arguments"]
            for argument in arguments:
                if argument[0] == "ARGM-TPC" or argument[0] == "A1" or argument[0] == "A2" or "A0" in argument[0]:
                    final_srl = argument[1]
                    flag = True
                    for stopword in stopwords:
                        if stopword in argument[1]:
                            flag = False
                            final_srl = final_srl.replace(stopword, " ")
                    if flag and len(final_srl) > 1:
                        arguments_list[final_srl] = 1
                    elif flag is False:
                        srl_list = final_srl.split(" ")
                        for srl in srl_list:
                            if len(srl) > 1:
                                arguments_list[srl] = 1
    return list(arguments_list)


def simi_levenshtein(str1, str2):
    # 越大越相似
    return normalized_levenshtein.similarity(str1, str2)


def simi_sentence_tranformers(str1, str2):
    # 越大越相似
    emb1 = model.encode(str1)
    emb2 = model.encode(str2)
    cos_sim = util.cos_sim(emb1, emb2)
    return cos_sim


def find_qa(user_query, qa_list, simi_levenshtein_threshold):
    keywords_user = keyword_extraction(user_query)
    srl_user = srl_extraction(user_query)
    list_user = list(set(keywords_user + srl_user))

    # 关键词筛选
    list1 = []
    for qa in qa_list:
        flag = False
        query_word = qa[3]
        for x in list_user:
            if x in query_word:
                list1.append(qa)
                break

    # 编辑距离筛选
    list2 = []
    sorted_list2 = []
    list2_levenshtein = []
    for qa in list1:
        q = qa[0]
        levenshtein = simi_levenshtein(q, user_query)
        if levenshtein > simi_levenshtein_threshold:
            list2.append(qa)
            list2_levenshtein.append(levenshtein)
    sorted_id2 = sorted(range(len(list2_levenshtein)), key=lambda k: list2_levenshtein[k], reverse=True)
    for id2 in sorted_id2:
        sorted_list2.append(list2[id2])

    # sentence transformers相似度筛选
    list3 = []
    sorted_list3 = []
    list3_sentence_tran = []
    for qa in list2:
        q = qa[0]
        sentence_tran = simi_sentence_tranformers(q, user_query).tolist()[0][0]
        list3.append(qa)
        list3_sentence_tran.append(sentence_tran)
    sorted_id3 = sorted(range(len(list3_sentence_tran)), key=lambda k: list3_sentence_tran[k], reverse=True)
    for id3 in sorted_id3:
        sorted_list3.append(list3[id3])
    return list1, sorted_list2, sorted_list3


query_list = []
with open('1.txt', 'r', encoding="utf-8") as f:
    for line in f.readlines():
        text_list = line.strip().split("\t")
        if len(text_list) == 4:
            query_list.append(text_list)

user_q = "无法连接服务的解决办法"
list_keyword, list_levenshtein, list_sentence_tranformers = find_qa(user_q, query_list, 0.5)

if len(list_sentence_tranformers) == 0:
    if len(list_levenshtein) == 0:
        if len(list_keyword) == 0:
            print("no result")
        else:
            print(list_keyword)
    else:
        print(list_levenshtein)
else:
    print(list_sentence_tranformers)

'''
# 便利所有的问题，抽取问题关键词
path = r"1.txt"
resultw = open(path, "w", encoding="utf-8")
df = pd.read_excel('1.xlsx')
df.fillna('', inplace=True)
for line in df.values:
    if len(line) == 3:
        query = line[0]
        answer1 = ch_text_pro1(line[1])
        answer2 = ch_text_pro1(str(line[2]))
        keywords = keyword_extraction(query)
        srl = srl_extraction(query)
        new_list = list(set(keywords + srl))
        resultw.write("\t".join([query, answer1, answer2, " ".join(new_list)])+"\n")

        # query_type = line[1]
        # answer = ch_text_pro1(line[2])
        # answer_people = ch_text_pro1(line[3])
        # reference = line[4].replace("\n", ", ")
        # keywords = keyword_extraction(query)
        # srl = srl_extraction(query)
        # new_list = list(set(keywords + srl))
        # resultw.write("\t".join([query, query_type, answer, answer_people, reference, " ".join(new_list)])+"\n")
resultw.close()
'''

