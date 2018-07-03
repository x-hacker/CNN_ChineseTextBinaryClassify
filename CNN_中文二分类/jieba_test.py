# coding:utf-8
import numpy as py
y_test = [1,0,0,0,1]
y=[]
for i in y_test:
	if i==0:
		y.append("交通")
	else:
		y.append("计算机")

print y[0],y[1],y[2]


import jieba  
"""
jieba的中文github地址：https://github.com/fxsjy/jieba
"""
x1='''

    三个臭皮匠顶个诸葛亮，以此类推，如果能把一个人跟另外100万人的大脑连接起来，就会诞生“超级大脑”。正因如此，现在才出现了好几家公司争相开发脑机界面，希望把人的思维与机器连接起来。如果能够率先将笔记本电脑的功能植入你的大脑，就将为人们开辟一条道路，使之得以随意通过无缝渠道与任何人（甚至任何东西）交换信息。目前有两位IT行业的大佬都在参与这场角逐，他们分别是特斯拉创始人埃隆·马斯克（Elon Musk）和Facebook创始人马克·扎克伯格（Mark Zuckerberg）。他们两人的项目分别名为Neuralink和Building 8。而据知情人士透露，这两个项目都需要对大脑进行外科手术。然而，还有一些没有那么野心勃勃的微创方式，也可以解决脑机界面问题。只需要把脑电波的数据转化成简单的指令，然后由应用或设备进行处理即可。一家名为Nuro的创业公司就采取了这种方式。他们希望借助自己的软件平台，让那些因为严重受伤或疾病而丧失交流能力的人恢复这种能力。
    '''
x2="本期企鹅评测团产品——华为MateBook X Pro笔记本电脑。作者是一名普通公务员，同时又是一名数码发烧友，多年来一直沉迷于各种新潮的数码产品，工作以后也不忘通过数码产品提升工作效率。随着笔记本电脑市场竞争的日益激烈，再加上硬件性能不断提升，越来越多的非游戏用户选择使用更加方便携带的超极本，各大厂商自然也是迎合用户需求，推出外观更加靓丽、身材更加小巧、功能更加丰富的超极本。"
seg_list = jieba.cut(x2.strip(), cut_all=True)  
# seg_list这是一个可循环的对象  
print("Full Mode: " + " ".join(seg_list))  # 全模式  

seg_list = jieba.cut("我来到北京清华大学,工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作", cut_all = False)  
print("Precise Mode: " + " ".join(seg_list))  #精确模式，默认状态下也是精确模式  

seg_list=jieba.cut_for_search("我来到北京清华大学,工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")  
print "搜索模式："," ".join(seg_list)

mcase = {'a': 10, 'b': 34}
mcase_frequency = {v: k for k, v in mcase.items()}
print mcase_frequency