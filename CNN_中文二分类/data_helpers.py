# coding:utf-8
import numpy as np
import re
import itertools
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 剔除英文的符号
def clean_str(string):
   
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(jisuanji_data_file, jiaotong_data_file):
    """
    加载二分类训练数据，为数据打上标签
    (X,[0,0])
    X = 【 日 期 】19960104 【 版 号 】1 【 标 题 】合巢芜高速公路巢芜段竣工 【 作 者 】彭建中 【 正 文 】 安徽合（肥）巢（湖）芜（湖）高速公路巢芜段日前竣工通车并投入营运。合巢芜 高速公路是国家规划的京福综合运输网的重要干线路段，是交通部确定１９９５年建成 的全国１０条重点公路之一。该条高速公路正线长８８公里。（彭建中）
    Y = 交通
    
    0:交通---> [1,0]

    1:计算机--->[0,1]
    
    (X,Y)

    """
    
    jisuanji_examples = list(open(jisuanji_data_file, "r").readlines())
    jisuanji_examples = [s.strip() for s in jisuanji_examples]
    jiaotong_exampless = list(open(jiaotong_data_file, "r").readlines())
    jiaotong_exampless = [s.strip() for s in jiaotong_exampless]
    x_text = jisuanji_examples + jiaotong_exampless
    
    # 适用于英文
    # x_text = [clean_str(sent) for sent in x_text]

    x_text = [sent for sent in x_text]
    # 定义类别标签 ，格式为one-hot的形式: y=1--->[0,1]
    positive_labels = [[0, 1] for _ in jisuanji_examples]
    # print positive_labels[1:3]
    negative_labels = [[1, 0] for _ in jiaotong_exampless]
    y = np.concatenate([positive_labels, negative_labels], 0)
    """
    print y
    [[0 1]
     [0 1]
     [0 1]
     ..., 
     [1 0]
     [1 0]
     [1 0]]
    print y.shape
    (10662, 2)
    """
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    利用迭代器从训练数据的回去某一个batch的数据
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # 每回合打乱顺序
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# 测试代码用的
if __name__ == '__main__':
    jisuanji_data_file = './fenci/jisuanji200.txt'
    jiaotong_data_file = './fenci/jiaotong214.txt'
    load_data_and_labels(jisuanji_data_file, jiaotong_data_file)








