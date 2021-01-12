# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/5/12 22:19 
# @Author: Zhang Cong

import random
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def generate_train_data(input_file_path, output_file_path):
    '''
    对原始csv数据进行格式转换，构建训练数据集
    :param input_file_path: 原始数据路径
    :param output_file_path: 构建完成的训练数据路径
    :return: 将数据存储至本地
    '''
    logging.info('Start get all sentence ...')
    # 获取全部句子集
    all_sentence = []
    for line in tqdm(open(input_file_path, encoding='utf-8')):
        line = line.replace('\n', '').split('\t')
        if line[2] == 'label':      # 跳过首行
            continue
        sentence_1 = str(line[0]).replace('\t', '')     # 句子1
        sentence_2 = str(line[1]).replace('\t', '')     # 句子2
        all_sentence.append(sentence_1)
        all_sentence.append(sentence_2)
    # 去重
    all_sentence = list(set(all_sentence))

    logging.info('Start generate dataset ...')
    # 构建训练数据集 [query, pos, neg_1, neg_2, neg_3, neg_4]
    output_file = open(output_file_path, mode='w', encoding='utf-8')
    for line in tqdm(open(input_file_path, encoding='utf-8')):
        line = line.replace('\n', '').split('\t')
        if line[2] == 'label':      # 跳过首行
            continue
        sentence_list = []
        sentence_1 = str(line[0]).replace('\t', '')
        sentence_2 = str(line[1]).replace('\t', '')
        sentence_list.append(sentence_1)    # 句子1
        sentence_list.append(sentence_2)    # 句子2
        label = line[2]                     # 标签

        if int(label)==1:       # 如果标签为1，则保留此句子对，并随机负采样得到4个负例
            while len(sentence_list)<6:         # [query, pos, neg_1, neg_2, neg_3, neg_4]
                index = random.randint(0, len(all_sentence)-1)      # 随机索引
                if all_sentence[index] not in sentence_list:        # 如果不重复，则加入
                    sentence_list.append(all_sentence[index])
            output_file.write('\t'.join(sentence_list) + '\n')
    output_file.close()
    logging.info('Finishied generate dataset ...')


def generate_test_data(input_file_path, output_file_path):
    '''
    对原始csv数据进行格式转换，构建测试数据集
    :param input_file_path: 原始数据路径
    :param output_file_path: 构建完成的训练数据路径
    :return: 将数据存储至本地
    '''
    logging.info('Start get all sentence ...')
    output_file = open(output_file_path, mode='w', encoding='utf-8')
    for line in tqdm(open(input_file_path, encoding='utf-8')):
        line = line.replace('\n', '').split('\t')
        if line[2] == 'label':      # 跳过首行
            continue
        sentence_1 = str(line[0]).replace('\t', '')     # 句子1
        sentence_2 = str(line[1]).replace('\t', '')     # 句子2
        label = line[2]                                 # 标签
        output_file.write(sentence_1 + '\t' + sentence_2 + '\t' + label + '\n')


def check_data(input_file_path):
    '''
    统计数据分布情况，检查数据集0/1分布是否均衡
    :param input_file_path: 数据路径
    :return:
    '''
    count = 0
    for line in tqdm(open(input_file_path, encoding='utf-8')):
        line = line.replace('\n', '').split('\t')
        if line[2] == 'label':
            continue
        if int(line[2]) == 1:
            count += 1
    print(count)


if __name__ == '__main__':

    # 统计数据分布情况
    file_path = './data/lcqmc/lcqmc_train.tsv'
    check_data(file_path)

    # 构建训练数据集
    input_file_path = './data/lcqmc/lcqmc_train.tsv'
    output_file_path = './data/train.txt'
    generate_train_data(input_file_path, output_file_path)
    logging.info('Success generate train.txt')

    # 构建验证数据集
    input_file_path = './data/lcqmc/lcqmc_dev.tsv'
    output_file_path = './data/dev.txt'
    generate_test_data(input_file_path, output_file_path)
    logging.info('Success generate dev.txt')

    # 构建测试数据集
    # input_file_path = './data/lcqmc/lcqmc_test.tsv'
    # output_file_path = './data/test.txt'
    # generate_test_data(input_file_path, output_file_path)
    # logging.info('Success generate test.txt')

