#引入包
import random
import jieba
import pandas as pd
#指定文件目录
dir = "fenci"
#指定语料
stop_words = "".join([dir,'stopwords.txt'])
laogong = "".join([dir,'beilaogongda.csv'])  #被老公打
laopo = "".join([dir,'beilaopoda.csv'])  #被老婆打
erzi = "".join([dir,'beierzida.csv'])   #被儿子打
nver = "".join([dir,'beinverda.csv'])    #被女儿打
#加载停用词
stopwords=pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
#加载语料
laogong_df = pd.read_csv(laogong, encoding='utf-8', sep=',')
laopo_df = pd.read_csv(laopo, encoding='utf-8', sep=',')
erzi_df = pd.read_csv(erzi, encoding='utf-8', sep=',')
nver_df = pd.read_csv(nver, encoding='utf-8', sep=',')
#删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
#转换
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()
#定义分词和打标签函数preprocess_text
#参数content_lines即为上面转换的list
#参数sentences是定义的空list，用来储存打标签之后的数据
#参数category 是类型标签
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
            segs = list(filter(lambda x:len(x)>1, segs)) #长度为1的字符
            segs = list(filter(lambda x:x not in stopwords, segs)) #去掉停用词
            sentences.append((" ".join(segs), category))# 打标签
        except Exception:
            print(line)
            continue 
#调用函数、生成训练数据
sentences = []
preprocess_text(laogong, sentences, 'laogong')
preprocess_text(laopo, sentences, 'laopo')
preprocess_text(erzi, sentences, 'erzi')
preprocess_text(nver, sentences, 'nver')

#打散数据，生成更可靠的训练集
random.shuffle(sentences)

#控制台输出前10条数据，观察一下
for sentence in sentences[:10]:
    print(sentence[0], sentence[1])
#用sk-learn对数据切分，分成训练集和测试集
from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

#抽取特征，我们对文本抽取词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
    max_features=4000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)
#用朴素贝叶斯算法进行模型训练
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
#对结果进行评分
print(classifier.score(vec.transform(x_test), y_test))