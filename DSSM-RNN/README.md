## DSSM-LSTM (Semantic modelling with long-short-term memory for information retrieval)


### 数据集：
#### LCQMC (http://icrc.hitsz.edu.cn/info/1037/1146.htm) 数据集版权保护，本项目不提供，请自行下载或替换其他数据集进行试验


### 数据形式：
#### query \t pos \t neg_1 \t neg_2 \t neg_3 \t neg_4
    其中pos为正例，neg_1、neg_2、neg_3、neg_4为随机负采样得到负例


### 文件解释
* main.py —— 主文件
* model.py —— 模型结构
* config.py —— 配置参数
* Data_Generate.py —— 数据集处理脚本
* /data —— 数据存放文件夹
* /save_model —— 模型存储文件夹


### 模型结构
![avatar](./DSSM-LSTM.png)
* 模型是在DSSM-CNN上的进一步改进，由于卷积无法捕捉单词之间的相关性，很明显更就要想到使用RNN解决这个问题。


### 参考资料
* Semantic modelling with long-short-term memory for information retrieval (https://arxiv.org/pdf/1412.6629.pdf)
* https://blog.csdn.net/sxf1061926959/article/details/89366526

