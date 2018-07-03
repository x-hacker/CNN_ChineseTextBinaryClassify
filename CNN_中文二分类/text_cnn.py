# coding:utf-8
import tensorflow as tf
import numpy as np

class TextCNN(object):
    '''
    sequence_length: 句子的长度，我们把所有的句子都填充成了相同的长度(该数据集是59)。
    num_classes: 输出层的类别数，我们这个例子是2(正向和负向)。
    vocab_size: 我们词汇表的大小。定义 embedding 层的大小的时候需要这个参数，embedding层的形状是[vocabulary_size, embedding_size]。
    embedding_size: 嵌入的维度。
    filter_sizes: 我们想要 convolutional filters 覆盖的words的个数，对于每个size，我们会有 num_filters 个 filters。比如 [3,4,5] 表示我们有分别滑过3，4，5个 words 的 filters，总共是3 * num_filters 个 filters。
    num_filters: 每一个filter size的filters数量(见上面)
    l2_reg_lambda:正则化处理时L2的参数
    '''
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 定义占位符，在dropout层保持的神经元的数量也是网络的输入，因为我们可以只在训练过程中启用dropout，而评估模型的时候不启用。
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # 定义常量，Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Embedding layer，强制操作运行在CPU上。 如果有GPU，TensorFlow 会默认尝试把操作运行在GPU上，但是embedding实现目前没有GPU支持，如果使用GPU会报错。
        # 这个 scope 添加所有的操作到一个叫做“embedding”的高阶节点，使得在TensorBoard可视化你的网络时，你可以得到一个好的层次
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                # 它将词汇表的词索引映射到低维向量表示。它基本上是我们从数据中学习到的lookup table
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # 创建实际的embedding操作。embedding操作的结果是形状为 [None, sequence_length, embedding_size] 的3维张量积
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # print "^^^^^^^embedded_chars^^^^^^",self.embedded_chars.get_shape()
            # (?, 56, 128)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # print "^^^^^^^embedded_chars_expanded^^^^^^",self.embedded_chars_expanded.get_shape()
            # (?, 56, 128, 1)

        # Create a convolution + maxpool layer for each filter size
        '''
        现在我们准备构建convolutional layers，然后进行max-pooling。注意我们使用不同大小的filters。
        因为每个卷积会产生不同形状的张量积，我们需要通过他们迭代，为每一个卷积构建一个卷积层，
        然后把结果合并为一个大的特征向量。
        这里，W是filter矩阵，h是对卷积输出应用非线性的结果(就是说卷积层用的是relu激活函数)，
        每个filter都在整个embedding上滑动，但是覆盖的words个数不同。”VALID” 填充意味着我们在句子上滑动filter没有填充边缘，
        做的是narrow convolution，可以得到形状为 [1, sequence_length - filter_size + 1, 1, 1]的输出。
        对特定的filter大小的输出进行max-pooling得到的是形状为[batch_size, 1, 1, num_filters]的张量积
        ，这基本是一个特征向量，最后的维度对应特征。一旦我们从每个filter size得到了所有的pool了的输出张量积，
        我们可以将它们结合在一起形成一个长的特征向量，形状是[batch_size, num_filters_total]。

        '''
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 定义参数，也就是模型的参数变量
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                """
                TensorFlow卷积conv2d操作接收的参数说明：
                self.embedded_chars_expanded：一个4维的张量，维度分别代表batch, width, height 和 channel
                W:权重
                strides：步长，是一个四维的张量[1, 1, 1, 1]，第一位与第四位固定为1，第二第三为长宽上的步长
                这里都是设为1
                padding：选择是否填充0，TensorFlow提供俩个选项，SAME、VAILD
                """
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # 非线性激活
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大值池化，h是卷积的结果，是一个四维矩阵，ksize是过滤器的尺寸，是一个四维数组，第一位第四位必须是1，第二位是长度，这里为卷积后的长度，第三位是宽度，这里为1
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 在 tf.reshape中使用-1是告诉TensorFlow把维度展平，作为全连接层的输入
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout，以概率1-dropout_keep_prob，随机丢弃一些节点
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        '''
            使用经过max-pooling (with dropout applied)的特征向量，我们可以通过做矩阵乘积，然后选择得分最高的类别进行预测。
            我们也可以应用softmax函数把原始的分数转化为规范化的概率，但是这不会改变我们最终的预测结果。
        '''
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # tf.nn.xw_plus_b 是进行 Wx+b 矩阵乘积的方便形式
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 定义损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
