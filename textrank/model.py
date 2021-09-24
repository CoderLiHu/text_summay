import pandas as pd
import numpy as np
import jieba
import re
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from tqdm import tqdm


# 构建数据清洗函数
def clean_sentence(sentence):
    # 1.将sentence按照 '|' 进行分句，只提取技师说的话
    sub_jishi = []
    sub = sentence.split('|')
    # print(sub)

    # 遍历每个句子
    for i in range(len(sub)):
        # print(sub[i])
        # 如果不是以句号结尾的，增加一个句号
        if not sub[i].endswith('。'):
            sub[i] += '。'
        # 只提取技师说的话
        if sub[i].startswith('技师'):
            sub_jishi.append(sub[i])

    # 拼接成字符串返回
    sentence = ''.join(sub_jishi)

    # 第二步中添加两个处理，利用正则表达式re工具
    # 2、删除1.，2.,3.这些标题
    r = re.compile("\D(\d\.)\D")
    sentence = r.sub('', sentence)

    # 3、删除一些无关紧要的词，语气助词
    r = re.compile(r'车主说|技师说|语音|图片|呢|吧|哈|啊|啦')
    sentence = r.sub("", sentence)

    # 第三步中添加的4个处理
    # 4. 删除带括号的 进口 海外
    r = re.compile(r"[(（]进口[)）]|\(海外\)")
    sentence = r.sub("", sentence)

    # 5. 删除除了汉字数字字母和，！？。.- 以外的字符
    r = re.compile("[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")

    # 6. 半角变为全角
    sentence = sentence.replace(",", "，")
    sentence = sentence.replace("!", "！")
    sentence = sentence.replace("?", "？")

    # 7. 问号叹号变为句号
    sentence = sentence.replace("？", "。")
    sentence = sentence.replace("！", "。")
    sentence = r.sub("", sentence)

    # 第四步添加的删除特定位置的特定字符
    # 8. 删除句子开头的逗号
    if sentence.startswith('，'):
        sentence = sentence[1:]

    return sentence


if __name__ == "__main__":
    # # 读取数据，并指定格式为 utf-8
    # df = pd.read_csv('./data/train.csv', engine='python', encoding='utf-8')
    # texts = df['Dialogue'].tolist()
    # print('预处理前的第一条句子：', texts[0])
    # print('*****************************')
    #
    # # 进行数据预处理
    # res = clean_sentence(texts[0])
    # print("预处理后的句子为：", res)

    # df = pd.read_csv('./data/train.csv', engine='python', encoding='utf-8')
    # print(len(df))
    # df = df.dropna(axis=0, subset=['Dialogue'])
    # print(len(df))
    # texts = df['Dialogue'].tolist()
    # preprocessed_list = []
    # for i in tqdm(range(len(texts))):
    #     # print(i)
    #     text = clean_sentence(texts[i])
    #     preprocessed_list.append(text)
    # df['PreprocessedDialogue'] = preprocessed_list
    # df.to_csv('./data/preprocessed_dialogue_train.csv', index=None, sep=',')


    # 读取预处理后的数据
    df = pd.read_csv('data/preprocessed_dialogue_train.csv', engine='python', encoding='utf-8')
    texts = df['PreprocessedDialogue'].tolist()
    # 初始化结果存放的列表
    results = []
    # 初始化textrank4zh类对象
    tr4s = TextRank4Sentence()

    # 循环遍历整个测试集，texts是处理后的对话数据
    for i in tqdm(range(len(texts))):
        text = texts[i]
        # 直接调用分析函数
        tr4s.analyze(text=text, lower=True, source='all_filters')
        result = ''

        # 直接调用函数获取关键语句
        # num=3: 获取重要性最高的3个句子.
        # sentence_min_len=2: 句子的长度最小等于2.
        for item in tr4s.get_key_sentences(num=3, sentence_min_len=2):
            result += item.sentence
            result += '。'

        results.append(result)

        # 间隔100次打印结果
        # if (i + 1) % 100 == 0:
        #     print(i + 1, result)

    print('result length: ', len(results))

    # 将结果保存到文件中
    # 保存结果
    df['Prediction'] = results

    # 提取ID, Report, 和预测结果这3列
    df = df[['QID', 'Report', 'Prediction']]

    # 保存结果，这里自动生成一个结果名
    df.to_csv('./data/textrank_result_.csv', index=None, sep=',')

    # 将空行置换为随时联系, 文件保存格式指定为utf-8
    df = pd.read_csv('data/textrank_result_.csv', engine='python', encoding='utf-8')
    df = df.fillna('随时联系。')

    # 将处理后的文件保存起来
    df.to_csv('./data/textrank_result_final_.csv', index=None, sep=',')

