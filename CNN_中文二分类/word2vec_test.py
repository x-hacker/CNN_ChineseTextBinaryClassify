# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
# python版的word2vec是包含在gensim工具包里面的，所以请用 pip install gensim 即可
import gensim.models.word2vec as w2v 
import logging,gensim,os
import multiprocessing 

# 显示现在可用的内核
print multiprocessing.cpu_count()
model_file_name = 'LW_w2c100.txt'  

#模型训练，生成词向量  
print "开始训练！"
sentences = w2v.LineSentence('./file/LW_01cut.txt')  
# sentences = w2v.LineSentence('WK_LW_cut.txt')  
model = w2v.Word2Vec(sentences, size=100, window=7, min_count=5,workers=multiprocessing.cpu_count())   


# 第一种
# model.save(model_file_name)
# 第二种、保存为字典形式的词向量模型，binary=0才可以，否则中文会产生乱码
model.save_word2vec_format(model_file_name, binary=0)  

print 'ok'
print "======================下面是进行比较============================"

print "======与‘卫星’相关========="
for x in model.most_similar(u"卫星",topn=7):
    print x[0],x[1]


print "======与‘遥感’相关========="
for x in model.most_similar(u"遥感",topn=7):
    print x[0],x[1]


# [Finished in 46.7s]
# wiki-400M[Finished in 875.6s
# size100 [Finished in 625.8s]
# '''

'''
model = w2v.Word2Vec.load('lw_mac_size100.bin')

a = "中国"
print model[u'%s' %(a)].shape
print model.wv[u'%s' %(a)]
# for k in model.similar_by_word([u"中国"],topn=3):
#     print k[0],k[1]
print "======================下面是进行比较============================"

print "======与‘卫星’相关========="
for x in model.most_similar(u"卫星",topn=7):
    print x[0],x[1]


print "======与‘遥感’相关========="
for x in model.most_similar(u"遥感",topn=7):
    print x[0],x[1]
'''

"""
1102实验结果：
======与‘卫星’相关=========
北斗 0.729344725609
导航 0.713463664055
测高 0.710172533989
全球定位系统 0.708398461342
云图 0.695049464703
定位系统 0.677260696888
全球卫星 0.672543406487


======与‘遥感’相关=========
遥感信息 0.775688707829
信息提取 0.768172860146
解译 0.752809762955
影像 0.75159060955
SAR 0.735499620438
遥感技术 0.711817026138
MODIS 0.692555189133
[Finished in 3.1s]

"""


"""
======================下面是进行比较1130[这个是使用关键字字典以及去除停用词、且只是用论文数据集训练的词向量模型]============================
======与‘卫星’相关=========
GPS 0.840189158916
TEC 0.742097973824
观测数据 0.730939686298
宽频带 0.722171843052
接收 0.716634392738
台网 0.69700717926
自由空气异常 0.693187355995
======与‘遥感’相关=========
遥感数据 0.864192664623
解译 0.829362094402
信息提取 0.829102158546
数字高程模型 0.828495025635
遥感图像 0.825434327126
遥感影像 0.812215149403
卫星遥感 0.800514996052
[Finished in 130.9s]
"""