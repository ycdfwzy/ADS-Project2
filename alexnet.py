# Created by NaXin

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from datetime import datetime
import math
import sys
import time
from six.moves import xrange
import tensorflow as tf
import numpy as np

# 输出网络参数
def print_actications(t):
    print(t.op.name, '', t.get_shape().as_list())
    return

# for test
# batch_size=32
# num_batches=100
# 创建一层卷积网络
# def AlexNetConv_new(images,keep_prob):
#     """Build the Alexnet model
#     参数：
#     训练图像集
#     返回：
#     pool5：卷积层的最后一个输出
#     paras：得到的每一卷积层的weights和biases
#     """
#     # layer 1
#     conv1 = createConv(images, [11,11,3,64], [1,4,4,1], 64, 'conv1', 'SAME')
#     lrn1 = createLrn(conv1, 1e-4, 0.75, 2, 2.0)
#     pool1 = createMaxPool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], name='pool1')
#
#     # layer 2
#     conv2 = createConv(pool1, [5,5,64,192], [1,1,1,1], 192, 'conv2', 'SAME')
#     lrn2 = createLrn(conv2, 1e-4, 0.75, 2, 2.0)
#     pool2 = createMaxPool(lrn2, [1,3,3,1], [1,2,2,1], 'pool2')
#
#     # layer 3
#     conv3 = createConv(pool2, [3,3,192,384], [1,1,1,1], 384, 'conv3', 'SAME')
#
#     # layer 4
#     conv4 = createConv(conv3, [3,3,384,256], [1,1,1,1], 256, 'conv4', 'SAME')
#
#     # layer 5
#     conv5 = createConv(conv4, [3,3,256,256], [1,1,1,1], 256, 'conv5', 'SAME')
#     pool5 = createMaxPool(conv5, [1,3,3,1], [1,2,2,1], 'pool5')
#
#     # layer 6
#     flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
#     # print(flattened.shape)
#     fc6 = createFullConnect(flattened, 6 * 6 * 256, 4096, name='fc6')
#     dropout6 = createDropout(fc6, keep_prob)
#     # print(fc6.shape)
#
#     # layer 7
#     fc7 = createFullConnect(dropout6, 4096, 4096, name='fc7')
#     dropout7 = createDropout(fc7,keep_prob)
#     print(dropout7.shape)
#     #return pool5

def createConv(input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups = 1):
    input_channels = int(input.get_shape()[-1])
    # 定义卷积核
    convolve = lambda i,ker: tf.nn.conv2d(i, ker, strides=[1,stride_y,stride_x,1], padding=padding)

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(input, kernel)
        else:
            input_group = tf.split(axis=3, num_or_size_splits=groups, value=input)
            weight_group = tf.split(axis=3, num_or_size_splits=groups, value=kernel)
            output_groups = [convolve(i,k) for i,k in zip(input_group,weight_group)]

            conv = tf.concat(axis=3, values=output_groups)

        # 设置偏置
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # 使用relu函数作为激活函数
        conv_ans = tf.nn.relu(bias, name = scope.name)
        # 输出网络信息
        # print_actications(conv_ans)

    return conv_ans

def createLrn(input, depth_radius, alpha, beta, name, bias=1.0):
    with tf.name_scope('lrn1') as scope:
        lrn = tf.nn.local_response_normalization(input, alpha=alpha, beta=beta, depth_radius=depth_radius, bias=bias)
    return lrn

def createMaxPool(input, filter_height, filter_width, stride_y, stride_x, name, padding='VALID'):
    pool = tf.nn.max_pool(input, ksize=[1,filter_height,filter_width,1], strides=[1,stride_y,stride_x,1],
                          padding=padding, name=name)
    # print_actications(pool)
    return pool

def createDropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

def createFullConnect(input, num_in, num_out, name, relu = True, tanh = False):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', shape=[num_out], trainable=True)

        act = tf.nn.xw_plus_b(input,weights,biases,name=scope.name)

        if relu:
            ans = tf.nn.relu(act)
            return ans
        elif tanh:
            ans1 = tf.nn.tanh(act)
            return ans1
        else:
            return act

class AlexNet():
    def __init__(self, images, keep_prob, num_classes, skip_layer, weights_path = 'DEFAULT'):
        '''
        INPUTS:
        :param images: tf.placeholder, input the images
        :param keep_prob: tf.placeholder, for the dropout rate
        :param num_classes: int, number of classes of the dataset
        :param skip_layer: list of strings, names of the layers you want to reinitialize
        :param weights_path: path string, to the pretrained weights(if blvc npy is not in the folder)
        '''
        self.images = images
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        # self.IS_TRAINING = is_training
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path
        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Build the Alexnet model
        参数：
        训练图像集
        返回：
        pool5：卷积层的最后一个输出
        paras：得到的每一卷积层的weights和biases
        """
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = createConv(self.images, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = createMaxPool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = createLrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = createConv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = createMaxPool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = createLrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = createConv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = createConv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = createConv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = createMaxPool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = createFullConnect(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = createDropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = createFullConnect(dropout6, 4096, 4096, name='fc7')
        dropout7 = createDropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = createFullConnect(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
        return

    def loadInitialWeights(self, session):
        # load the weight
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            # 在跳过的层中，说明不使用外部参数，则需要进行学习
            if op_name in self.SKIP_LAYER:
                continue
            with tf.variable_scope(op_name, reuse=True):
                # loop the list
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        bia = tf.get_variable('biases',trainable=False)
                        session.run(bia.assign(data))
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))
