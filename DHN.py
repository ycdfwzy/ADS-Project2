from alexnet import *
import random
from math import *

class DeepHashingNet():
    def __init__(self, images, keep_prob, hash_K, skip_layer, weights_path = 'DEFAULT'):
        '''
        INPUTS:
        :param images: tf.placeholder, input the images
        :param keep_prob: tf.placeholder, for the dropout rate
        :param num_classes: int, number of classes of the dataset
        :param skip_layer: list of strings, names of the layers you want to reinitialize
        :param weights_path: path string, to the pretrained weights(if blvc npy is not in the folder)
        '''
        self.images = images
        self.hash_K = hash_K
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
        #dropout7 = createDropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fch = createFullConnect(fc7, 4096, self.hash_K, relu=False, name='fch', tanh=True)
        return

    def loadInitialWeights(self, session):
        # load the weight
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            # 在跳过的层中，说明不使用外部参数，则需要进行学习
            if op_name in self.SKIP_LAYER or op_name == 'fc8':
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

# loss L
def pairwise_cross_entropy_loss(hash_code, oneHotLabel, alpha = 1, threshold = 15, class_num = 10, normed = False):
    # 获取相似度
    # similarity = tf.matmul(oneHotLabel, oneHotLabel, transpose_b=True)
    #
    # dot_product = tf.matmul(hash_code, hash_code, transpose_b=True)
    # exp_product = tf.exp(sigmoid_param*dot_product)
    #
    # mask_dot = dot_product > threshold # 值过大，直接使用dot
    # mask_exp = dot_product <= threshold # 值较小，使用exp
    #
    # dot_loss = dot_product * (1 - similarity)
    # exp_loss = tf.log(1 + exp_product) - (similarity * dot_product)
    #
    # loss = tf.reduce_sum(tf.boolean_mask(dot_loss, mask_dot)) + tf.reduce_sum(tf.boolean_mask(exp_loss, mask_exp))
    # size = tf.to_float(hash_code.shape[0])
    # # return loss/(size * size)
    # return loss

    label_ip = tf.cast(
        tf.matmul(oneHotLabel, tf.transpose(oneHotLabel)), tf.float32)
    s = tf.clip_by_value(label_ip, 0.0, 1.0)

    # compute balance param
    # s_t \in {-1, 1}
    s_t = tf.multiply(tf.add(s, tf.constant(-0.5)), tf.constant(2.0))
    sum_1 = tf.reduce_sum(s)
    sum_all = tf.reduce_sum(tf.abs(s_t))
    balance_param = tf.add(tf.abs(tf.add(s, tf.constant(-1.0))),
                           tf.multiply(tf.div(sum_all, sum_1), s))

    if normed:
        # ip = tf.clip_by_value(tf.matmul(u, tf.transpose(u)), -1.5e1, 1.5e1)
        ip_1 = tf.matmul(hash_code, tf.transpose(hash_code))

        def reduce_shaper(t):
            return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

        mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(hash_code)),
                                  reduce_shaper(tf.square(hash_code)), transpose_b=True))
        ip = tf.div(ip_1, mod_1)
    else:
        ip = tf.clip_by_value(tf.matmul(hash_code, tf.transpose(hash_code)), -1.5e1, 1.5e1)
    ones = tf.ones([tf.shape(hash_code)[0], tf.shape(hash_code)[0]])
    return tf.reduce_mean(tf.multiply(tf.log(ones + tf.exp(alpha * ip)) - s * alpha * ip, balance_param))

def quantization_loss(hash_code):
    size = tf.to_float(hash_code.shape[0])
    return tf.reduce_sum(-tf.log(tf.cosh(tf.abs(hash_code) - 1)))

# imageSize = 227
# images = tf.Variable(tf.random_normal([128, imageSize, imageSize, 3], dtype=tf.float32, stddev=1e-1))
# dhn = DeepHashingNet(images, 0.05,10, ['fc7', 'fch'])
# labels = []
# for i in range(128):
#     t = random.randint(0, 10)
#     la = []
#     for j in range(10):
#         if j != t: la.append(0.0)
#         else: la.append(1.0)
#     labels.append(la)
# # print(labels)
# labels_t = tf.Variable(tf.constant(labels), dtype=tf.float32)
#
# pair = pairwise_cross_entropy_loss(dhn.fch, labels_t)
# te = test(dhn.fch, labels_t)
# # loss = quantization_loss(dhn.fch)
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
#
# print(sess.run(pair))
# print(sess.run(te))
# print(dhn.fch)
