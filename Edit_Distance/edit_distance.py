# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/25 22:44 
# @Author: Zhang Cong

'''
    编辑距离
'''
def edit_distance(string_1, string_2):
    if len(string_1) == 0:      # 如果string_1长度为0，则返回string_2的长度为结果
        return len(string_2)
    if len(string_2) == 0:      # 如果string_2长度为0，则返回string_1的长度为结果
        return len(string_1)

    if string_1[0] == string_2[0]:  # 如果string_1和string_2的首字符相同，则同时去掉首字母，继续递归
        return edit_distance(string_1[1: ], string_2[1: ])
    else:
        return min(edit_distance(string_1[1: ], string_2) + 1,          # string_1去掉首字符
                   edit_distance(string_1[1: ], string_2[1: ]) + 1,     # string_1和string_2都去掉首字符
                   edit_distance(string_1, string_2[1: ]) + 1)          # string_2去掉首字符


if __name__ == '__main__':
    string_1 = 'abcde'
    string_2 = 'ac'
    num_step = edit_distance(string_1, string_2)
    score = 1 - (num_step/max(len(string_1), len(string_2)))
    print(score)