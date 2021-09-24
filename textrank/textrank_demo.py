import os
import sys
# 导入关键词工具包
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas
import numpy as np

# 关键词抽取函数
def keywords_extraction(text):
    # allow_speech_tags : 词性列表, 用于过滤某些词性的词
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])

    # text: 文本内容, 字符串
    # window: 窗口大小, int, 用来构造单词之间的边, 默认值为2
    # lower: 是否将英文文本转换为小写, 默认值为False
    # vertex_source: 选择使用words_no_filter, words_no_stop_words, words_all_filters中的>哪一个来构造pagerank对应的图中的节点
    #                默认值为'all_filters', 可选值为'no_filter', 'no_stop_words', 'all_filters'
    # edge_source: 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪>一个来构造pagerank对应的图中的节点之间的边
    #              默认值为'no_stop_words', 可选值为'no_filter', 'no_stop_words', 'all_filters', 边的构造要结合window参数
    # pagerank_config: pagerank算法参数配置, 阻尼系数为0.85

    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters',
                 edge_source='no_stop_words', pagerank_config={'alpha': 0.85, })

    # num: 返回关键词数量
    # word_min_len: 词的最小长度, 默认值为1
    keywords = tr4w.get_keywords(num=6, word_min_len=2)

    # 返回关键词
    return keywords

# 关键词短语的抽取函数
def keyphrases_extraction(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters',
                 edge_source='no_stop_words', pagerank_config={'alpha': 0.85, })

    # keywords_num:抽取关键词的数量
    # min_occur_num:关键短语在文中最小的出现次数
    keyphrases = tr4w.get_keyphrases(keywords_num=5, min_occur_num=1)

    # 返回关键短语结果
    return keyphrases

# 关键句抽取函数
def keysentences_extraction(text):
    tr4s = TextRank4Sentence()

    # text：文本内容字符串
    # lower: 是否将英文文本转换为小写，默认值为False
    # source: 默认值为 'all_filters'
    tr4s.analyze(text, lower=True, source='all_filters')

    # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
    keysentences = tr4s.get_key_sentences(num=3, sentence_min_len=6)

    # 返回关键句子
    return keysentences

# 基于jieba的TextRank算法
import jieba.analyse

# 构建jieba提取关键词的函数
def jieba_keywords_textrank(text):
    keywords = jieba.analyse.textrank(text, topK=6)
    return keywords

if __name__ == "__main__":
    text = "来源：中国科学报本报讯（记者肖洁）又有一位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院" \
           "国家天文台在京举行“周又元星”颁授仪式，" \
           "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济一堂。国家天文台党委书记、" \
           "副台长赵刚在致辞一开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
           "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站，" \
           "获得国际永久编号第120730号。2018年9月25日，经国家天文台申报，" \
           "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
           "正式将该小行星命名为“周又元星”。"

    #关键词抽取
    # keywords=keywords_extraction(text)
    # for word in keywords:
    #     print(word)

    # 关键短语抽取
    # keyphrases = keyphrases_extraction(text)
    # for phrase in keyphrases:
    #     print(phrase)

    #关键句抽取
    # keysentences=keysentences_extraction(text)
    # for sentence in keysentences:
    #     print(sentence)

    # jieba提取关键词
    # keywords = jieba_keywords_textrank(text)
    # print(keywords)