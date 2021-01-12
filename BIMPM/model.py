# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/7/16 22:58
# @Author: Zhang Cong

import tensorflow as tf
from config import Config
import tensorflow.contrib as contrib

class Model():
    def __init__(self):
        self.config = Config()                                                                                                   # 读取配置参数
        self.input_query_word = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-query-word")    # 输入query，One-Hot形式
        self.input_doc_word = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-doc-word")        # 输入doc，One-Hot形式
        self.input_query_char = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-query-char")    # 输入query，One-Hot形式
        self.input_doc_char = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name="input-doc-char")        # 输入doc，One-Hot形式
        self.input_label = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name="input-label")             # 输入 label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                                          # keep-prob

        # Embedding layer
        self.embedding_word = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding-word')
        self.embedding_char = tf.get_variable(shape=[self.config.char_size, self.config.embedding_dim], dtype=tf.float32, name='embedding-char')

        # 将词汇映射为向量形式 [batch_size, seq_length, embedding_dim]
        self.input_query_word_emb = tf.nn.embedding_lookup(params=self.embedding_word, ids=self.input_query_word, name='input-query-word-emb')
        self.input_doc_word_emb = tf.nn.embedding_lookup(params=self.embedding_word, ids=self.input_doc_word, name='input-doc-word-emb')

        # ----- Word Representation Layer -----
        # 将字符映射为向量形式 [batch_size, seq_length, embedding_dim]
        self.input_query_char_emb = tf.nn.embedding_lookup(params=self.embedding_char, ids=self.input_query_char, name='input-query-char-emb')
        self.input_doc_char_emb = tf.nn.embedding_lookup(params=self.embedding_char, ids=self.input_doc_char, name='input-doc-char-emb')

        # 将字符传入LSTM后作为char embedding
        input_query_char_emb = self.uni_directional_rnn(input_data=self.input_query_char_emb,
                                                             num_units=self.config.hidden_dim,
                                                             rnn_type=self.config.rnn_type,
                                                             scope='rnn-query-char')
        input_doc_char_emb = self.uni_directional_rnn(input_data=self.input_doc_char_emb,
                                                           num_units=self.config.hidden_dim,
                                                           rnn_type=self.config.rnn_type,
                                                           scope='rnn-doc-char')
        # 将生成的char embedding 与 word embedding进行拼接 [batch_size, seq_length, word_embediing + char_hidden_dim]
        self.query_embedding = tf.concat([input_query_char_emb, self.input_query_word_emb], axis=-1)
        self.doc_embedding = tf.concat([input_doc_char_emb, self.input_doc_word_emb], axis=-1)

        # dropout layer
        self.query_embedding = tf.nn.dropout(self.query_embedding, keep_prob=self.input_keep_prob)
        self.doc_embedding = tf.nn.dropout(self.doc_embedding, keep_prob=self.input_keep_prob)

        # ----- Context Representation Layer -----
        # 对query和doc向量进行Bi-LSTM处理
        query_fw, query_bw = self.bi_directional_rnn(input_data=self.query_embedding,
                                                     num_units=self.config.hidden_dim,
                                                     rnn_type=self.config.rnn_type,
                                                     scope='bi-rnn-query-char')
        doc_fw, doc_bw = self.bi_directional_rnn(input_data=self.doc_embedding,
                                                 num_units=self.config.hidden_dim,
                                                 rnn_type=self.config.rnn_type,
                                                 scope='bi-rnn-doc-char')

        # dropout layer
        query_fw = tf.nn.dropout(query_fw, keep_prob=self.input_keep_prob)
        query_bw = tf.nn.dropout(query_bw, keep_prob=self.input_keep_prob)
        doc_fw = tf.nn.dropout(doc_fw, keep_prob=self.input_keep_prob)
        doc_bw = tf.nn.dropout(doc_bw, keep_prob=self.input_keep_prob)

        # ----- Matching Layer -----
        # 1、Full-Matching
        # 这个匹配策略是对于query中Bi-LSTM的每个时间步与doc中Bi-LSTM的最后一个时间步计算相似度（既有前向也有后向），然后doc的每个
        # 时间步与query的最后一个时间步计算相似度
        w1 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w1')
        w2 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w2')
        query_full_fw = self.full_matching(query_fw, tf.expand_dims(doc_fw[:, -1, :], 1), w1)
        query_full_bw = self.full_matching(query_bw, tf.expand_dims(doc_bw[:, 0, :], 1), w2)
        doc_full_fw = self.full_matching(doc_fw, tf.expand_dims(query_fw[:, -1, :], 1), w1)
        doc_full_bw = self.full_matching(doc_bw, tf.expand_dims(query_bw[:, 0, :], 1), w2)

        # 2、Maxpooling-Matching
        # 这个匹配策略对于P中BiLSTM的每个时间步与Q中BiLSTM的每个时间步分别计算相似度，然后只返回最大的一个相似度
        w3 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w3')
        w4 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w4')
        max_fw = self.maxpool_matching(query_fw, doc_fw, w3)
        max_bw = self.maxpool_matching(query_bw, doc_bw, w4)

        # 3、Attentive-Matching
        # 这个匹配策略先计算P和Q中BiLSTM中每个时间步的cosine(传统的)相似度，生成一个相关性矩阵，然后用这个相关矩阵计算Q的加权求和（如果是P-->Q），
        # 最后用P的每个时间步分别于Q的加权求和计算相似度
        # 计算权重即相似度矩阵（普通Cosine）
        fw_cos = self.cosine(query_fw, doc_fw)
        bw_cos = self.cosine(query_bw, doc_bw)

        # 计算attentive vector 加权求和
        query_att_fw = tf.matmul(fw_cos, query_fw)
        query_att_bw = tf.matmul(bw_cos, query_bw)
        doc_att_fw = tf.matmul(fw_cos, doc_fw)
        doc_att_bw = tf.matmul(bw_cos, doc_bw)
        # 标准化，除以权重和
        query_mean_fw = tf.divide(query_att_fw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        query_mean_bw = tf.divide(query_att_bw, tf.reduce_sum(bw_cos, axis=2, keep_dims=True))
        doc_mean_fw = tf.divide(doc_att_fw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        doc_mean_bw = tf.divide(doc_att_bw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        # 计算match score
        w5 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w5')
        w6 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w6')
        query_att_mean_fw = self.full_matching(query_fw, query_mean_fw, w5)
        query_att_mean_bw = self.full_matching(query_bw, query_mean_bw, w6)
        doc_att_mean_fw = self.full_matching(doc_fw, doc_mean_fw, w5)
        doc_att_mean_bw = self.full_matching(doc_bw, doc_mean_bw, w6)

        # 4、Max-Attentive-Matching
        # 这个和上面的attentive-matching很像，只不过这里不再是加权求和了，而是直接用cosine最大的embedding作为attentive vector，
        # 然后P的每个时间步分别于最大相似度的embedding求多角度cosine相似度
        # 求cos最大的embedding
        query_max_fw = tf.reduce_max(query_att_fw, axis=2, keep_dims=True)
        query_max_bw = tf.reduce_max(query_att_bw, axis=2, keep_dims=True)
        doc_max_fw = tf.reduce_max(doc_att_fw, axis=2, keep_dims=True)
        doc_max_bw = tf.reduce_max(doc_att_bw, axis=2, keep_dims=True)
        # 计算match score
        w7 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w7')
        w8 = tf.get_variable(shape=[self.config.num_perspective, self.config.hidden_dim], dtype=tf.float32, name='w8')
        query_att_max_fw = self.full_matching(query_fw, query_max_fw, w7)
        query_att_max_bw = self.full_matching(query_bw, query_max_bw, w8)
        doc_att_max_fw = self.full_matching(doc_fw, doc_max_fw, w7)
        doc_att_max_bw = self.full_matching(doc_bw, doc_max_bw, w8)

        # 将以上四种相似度计算方式得出的结果进行拼接
        mv_query = tf.concat([query_full_fw, max_fw, query_att_mean_fw, query_att_max_fw, query_full_bw, max_bw, query_att_mean_bw, query_att_max_bw], axis=2)
        mv_doc = tf.concat([doc_full_fw, max_fw, doc_att_mean_fw, doc_att_max_fw, doc_full_bw, max_bw, doc_att_mean_bw, doc_att_max_bw], axis=2)
        # dropout layer
        mv_query  = tf.nn.dropout(mv_query, keep_prob=self.input_keep_prob)
        mv_doc = tf.nn.dropout(mv_doc, keep_prob=self.input_keep_prob)
        # 维度转换
        mv_query = tf.reshape(mv_query, [-1, mv_query.shape[1], mv_query.shape[2] * mv_query.shape[3]])
        mv_doc = tf.reshape(mv_doc, [-1, mv_doc.shape[1], mv_doc.shape[2] * mv_doc.shape[3]])

        # ----- Aggregation Layer -----
        # 采用Bi-LSTM对合并转换后的向量进行特征提取
        query_fw_final, query_bw_final = self.bi_directional_rnn(input_data=mv_query,
                                                                 num_units=self.config.hidden_dim,
                                                                 rnn_type=self.config.rnn_type,
                                                                 scope='bi-rnn-query-agg')
        doc_fw_final, doc_bw_final = self.bi_directional_rnn(input_data=mv_doc,
                                                             num_units=self.config.hidden_dim,
                                                             rnn_type=self.config.rnn_type,
                                                             scope='bi-rnn-doc-agg')
        # 将Bi-LSTM的结果进行拼接
        combine_emb = tf.concat((query_fw_final, query_bw_final, doc_fw_final, doc_bw_final), axis=2)
        combine_emb = tf.reshape(combine_emb, shape=[-1, combine_emb.shape[1] * combine_emb.shape[2]])
        combine_emb = tf.nn.dropout(combine_emb, keep_prob=self.input_keep_prob)

        # ----- Prediction Layer -----
        # 全连接层 1
        with tf.variable_scope('feed_foward_layer1'):
            inputs = tf.nn.dropout(combine_emb, self.input_keep_prob)
            outputs = tf.layers.dense(inputs=inputs,
                                      units=self.config.fc_hidden_dim_1,
                                      activation=tf.nn.relu,
                                      use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
        # 全连接层 2
        with tf.variable_scope('feed_foward_layer2'):
            inputs = tf.nn.dropout(outputs, self.input_keep_prob)
            outputs = tf.layers.dense(inputs=inputs,
                                      units=self.config.fc_hidden_dim_2,
                                      activation=tf.nn.relu,
                                      use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
        # 全连接层 3
        with tf.variable_scope('feed_foward_layer3'):
            inputs = tf.nn.dropout(outputs, self.input_keep_prob)
            self.logits = tf.layers.dense(inputs=inputs,
                                          units=self.config.num_classes,
                                          activation=tf.nn.relu,
                                          use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
        # 类别score
        self.score = tf.nn.softmax(self.logits, name='score')
        # 预测结果
        self.predict = tf.argmax(self.score, axis=1, name='predict')
        # 准确率
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_label, axis=1), self.predict), dtype=tf.float32),name='accuracy')
        # 结构化损失函数，交叉熵+L2正则化
        self.loss = tf.add(
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_label)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="loss")
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, name="optimizer").minimize(self.loss)



    def full_matching(self, metric, vec, w):
        '''
        1、Full-Matching相似度计算，metric中的每个时间步与vec进行相似度计算
        :param metric: 时间步矩阵 [batch_size, seq_length, hidden_dim]
        :param vec: 最后一个时间步输出向量 [batch_size, 1, hidden_dim]
        :param w: 权重矩阵 [num_perspective, hidden_dim]
        :return:
        '''
        w = tf.expand_dims(tf.expand_dims(w, 0), 2)                             # 构建多角度权重矩阵 [batch_size, 1, num_perspective, hidden_dim]
        metric = w * tf.stack([metric] * self.config.num_perspective, axis=1)   # 生成多角度metric向量
        vec = w * tf.stack([vec] * self.config.num_perspective, axis=1)         # 生成多角度vec向量
        # 进行Cosine计算
        m = tf.matmul(metric, tf.transpose(vec, [0, 1, 3, 2]))                              # metric与vec进行点乘（cos分子）
        n = tf.norm(metric, axis=3, keep_dims=True) * tf.norm(vec, axis=3, keep_dims=True)  # metric的L2范数与vec的L2范数相乘（cos分母）
        cosine = tf.transpose(tf.divide(m, n), [0, 2, 3, 1])          # 相除得到Cosine [batch_size, seq_length, 1, num_perspective]

        return cosine


    def maxpool_matching(self, v1, v2, w):
        '''
        2、Maxpooling-Matching相似度计算，v1中的每个时间步与v2中的每个时间步进行相似度计算
        :param v1: 时间步矩阵 [batch_size, seq_length, hidden_dim]
        :param v2: 时间步矩阵 [batch_size, seq_length, hidden_dim]
        :param w: 权重矩阵 [num_perspective, hidden_dim]
        :return:
        '''
        cosine = self.full_matching(v1, v2, w)      # full_matching相似度计算
        max_value = tf.reduce_max(cosine, axis=2, keep_dims=True)   # maxpooling
        return max_value


    def cosine(self, v1, v2):
        '''
        计算两个矩阵每个时间步的cos值
        :param v1: 时间步矩阵1
        :param v2: 时间步矩阵2
        :return:
        '''
        m = tf.matmul(v1, tf.transpose(v2, [0, 2, 1]))      # 矩阵v1和矩阵v2进行点乘（cos分子）
        n = tf.norm(v1, axis=2, keep_dims=True) * tf.norm(v2, axis=2, keep_dims=True)       # v1c的L2范数与v2的L2范数相乘（cos分母）
        cosine = tf.divide(m, n)                     # 相除得到Cosine
        return cosine


    def bi_directional_rnn(self, input_data, num_units, rnn_type, scope, reuse=False):
        '''
        构建双向RNN层，可选LSTM/GRU
        :param input_data: 输入时序数据
        :param rnn_type: RNN类型
        :param scope: 变量空间
        :param reuse: 是否重用变量
        :return:
        '''
        with tf.variable_scope(scope, reuse=reuse):
            cell_fw = self.get_rnn(rnn_type, num_units)
            cell_bw = self.get_rnn(rnn_type, num_units)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=input_data, dtype=tf.float32)
            # outputs = tf.concat(outputs, axis=2)
            return outputs


    def uni_directional_rnn(self, input_data, num_units, rnn_type, scope, reuse=False):
        '''
        构建单向RNN层，可选LSTM/GRU
        :param input_data: 输入时序数据
        :param rnn_type: RNN类型
        :param scope: 变量空间
        :param reuse: 是否重用变量
        :return:
        '''
        with tf.variable_scope(scope, reuse=reuse):
            cell = self.get_rnn(rnn_type, num_units)
            outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input_data, dtype=tf.float32)
            return outputs


    def get_rnn(self, rnn_type, num_units):
        '''
        根据rnn_type创建RNN层
        :param rnn_type: RNN类型
        :return:
        '''
        if rnn_type == 'lstm':
            cell = contrib.rnn.LSTMCell(num_units=num_units)
        else:
            cell = contrib.rnn.GRUCell(num_units=num_units)
        cell = contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=self.input_keep_prob)
        return cell


if __name__ == '__main__':
    Model()