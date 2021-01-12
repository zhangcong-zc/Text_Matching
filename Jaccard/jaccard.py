# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/25 22:50 
# @Author: Zhang Cong

'''
    Jaccard 相似度
'''


def jaccard(string_1, string_2):
    char_set_1 = set(string_1)
    char_set_2 = set(string_2)
    interaction = char_set_1.intersection(char_set_2)   # 取交集
    union = char_set_1.union(char_set_2)                # 取并集
    score = len(interaction)/len(union)                 # 计算score
    return score


if __name__ == '__main__':
    string_1 = 'abcdef'
    string_2 = 'ab'
    score = jaccard(string_1, string_2)
    print('Score: {}'.format(score))

    