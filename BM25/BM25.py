# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/7/20 21:50 
# @Author: Zhang Cong

import math
import jieba


class BM25(object):
    def __init__(self, docs):
        self.N = len(docs)  # 文本数量
        self.avgdl = sum([len(doc) for doc in docs])*1.0 / self.N   # 文本平均长度
        self.docs = docs
        self.f = []         # 每篇文档中每个词的出现次数
        self.df = {}        # 每个词及出现了该词的文档数量
        self.idf = {}       # 每个词的IDF值
        self.k1 = 1.5       # 调节参数K1
        self.b = 0.75       # 调节参数b
        self.init()


    def init(self):
        '''
        计算文档集每篇文档中每个词的出现次数、每个词及出现了该词的文档数量、每个词的IDF值
        :return:
        '''
        for doc in self.docs:
            tmp = {}
            # 统计当前文档中每个词的出现次数
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)  # 加入到全局记录中

            # 统计出现了当前词汇的文档数量
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1

        # 计算IDF值
        for k, v in self.df.items():
            self.idf[k] = math.log(self.N-v+0.5)-math.log(v+0.5)


    def get_score(self, query, index):
        '''
        计算输入的query和doc的相似度分数score
        :param doc: 输入的query
        :param index: 文档集中的文档索引
        :return:
        '''
        score = 0
        for word in query:
            # 如果是未登录词，则跳过
            if word not in self.f[index]:
                continue
            dl = len(self.docs[index])  # 当前文档长度
            # 计算相似度分数 IDF*R(q, d) 求和
            score += (self.idf[word] * self.f[index][word]*(self.k1+1)
                                        / (self.f[index][word] + self.k1 * (1 - self.b + self.b*dl/self.avgdl)))
        return score


    def similarity(self, query):
        '''
        输入query对文档集进行检索
        :param doc: 分词后的query list
        :return:
        '''
        scores = []
        for index in range(self.N):
            score = self.get_score(query, index)
            scores.append(score)
        return scores


if __name__ == '__main__':
    # 测试文本
    text = ['自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。',
            '它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。',
            '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。',
            '因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，',
            '所以它与语言学的研究有着密切的联系，但又有重要的区别。',
            '自然语言处理并不是一般地研究自然语言，',
            '而在于研制能有效地实现自然语言通信的计算机系统，',
            '特别是其中的软件系统。因而它是计算机科学的一部分。']

    doc = []
    for sentence in text:
        words = list(jieba.cut(sentence))
        doc.append(words)
    print(doc)
    s = BM25(doc)
    print(s.f)
    print(s.idf)
    print(s.similarity(['自然语言', '计算机科学', '领域', '人工智能', '领域']))