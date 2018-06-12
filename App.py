import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from DataGenerator import ImageDataGenerator
from DHN import *

train_path ='data/train'
var_path = 'data/test'

# learning params
learning_rate = 0.001
num_epochs = 10
batch_size = 128
lam = 0.2
Train = False
filewriter_path = os.path.join(os.getcwd(), "tmp\\dhn\\test")
checkpoint_path = os.path.join(os.getcwd(), "tmp\\dhn")

# Network params
dropout_rate = 0.5
hash_K = 48
num_classes = 10
train_layers = ['fc7', 'fch']

def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0 else '1'
        list_string_binary.append(str)
    return list_string_binary

def toString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        st = ''
        for j in range(bit_length):
            st += str(binary_like_values[i][j]) + ' '
        list_string_binary.append(st)
    return list_string_binary

def train():
    # 创建检查点文件夹
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    # TF place holder
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # build model
    model = DeepHashingNet(x, keep_prob, hash_K, train_layers)

    score = model.fch

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
    # var_list = tf.trainable_variables()

    with tf.name_scope('cross_ent'):
        # loss = pairwise_cross_entropy_loss(score, y, class_num=10) + lam * quantization_loss(score)
        loss = pairwise_cross_entropy_loss(score, y, class_num=10)+  tf.multiply(tf.Variable(lam),
                                                                                 tf.reduce_mean(tf.square(tf.subtract(tf.abs(score), tf.constant(1.0)))))
        # tf.Print(loss, [loss])
        # loss = pairwise_cross_entropy_loss(score, y, class_num=10)

    # Train op
    with tf.name_scope('train'):
        # 计算梯度
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # # 添加一堆summary
    # # Add gradients to summary
    # for gradient, var in gradients:
    #     tf.summary.histogram(var.name + '/gradient', gradient)
    # # Add the variables we train to the summary
    # for var in var_list:
    #     tf.summary.histogram(var.name, var)
    # # Add the loss to summary
    # tf.summary.scalar('loss_func', loss)

    # Evaluation op: Accuracy of the model
    # with tf.name_scope("accuracy"):
    #     correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Add the accuracy to the summary
    # tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    # merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Initialize the data generator seperately for the training and validation set
    train_generator = ImageDataGenerator(train_path,
                                         horizontal_flip=True, shuffle=True)
    val_generator = ImageDataGenerator(var_path, shuffle=False)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.dataSize / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.dataSize / batch_size).astype(np.int16)

    # Start Tensorflow session
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.loadInitialWeights(sess)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            step = 1

            while step < train_batches_per_epoch:

                # Get a batch of images and labels
                batch_xs, batch_ys = train_generator.next_batch(batch_size)

                # And run the training op
                sess.run(train_op, feed_dict={x: batch_xs,
                                              y: batch_ys,
                                              keep_prob: dropout_rate})
                print("{} Epoch number: {}".format(datetime.now(), step))
                # Generate summary with the current batch of data and write to file
                # if step % display_step == 0:
                #     s = sess.run(merged_summary, feed_dict={x: batch_xs,
                #                                             y: batch_ys,
                #                                             keep_prob: 1.})
                #     writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            # Validate the model on the entire validation set
            # print("{} Start validation".format(datetime.now()))
            # test_acc = 0.
            # test_count = 0
            # for _ in range(val_batches_per_epoch):
            #     batch_tx, batch_ty = val_generator.next_batch(batch_size)
            #     acc = sess.run(accuracy, feed_dict={x: batch_tx,
            #                                         y: batch_ty,
            #                                         keep_prob: 1.})
            #     test_acc += acc
            #     test_count += 1
            # test_acc /= test_count
            # print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

            # Reset the file pointer of the image data generator
            val_generator.reset()
            train_generator.reset()

            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

def getHash():
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='x')
    model = DeepHashingNet(x, 1, hash_K, [])
    fch = model.fch
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver(tf.all_variables())

    sess = tf.InteractiveSession()

    train_generator = ImageDataGenerator(train_path,
                                         horizontal_flip=False, shuffle=False)
    val_generator = ImageDataGenerator(var_path, shuffle=False)

    file_res = open('result_dhn.txt', 'w')
    # sys.stdout = file_res
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(checkpoint_path, ckpt_name))
        print('Loading success, global_step is %s' % global_step)

        k = 0
        for i in range(val_generator.dataSize // batch_size):
            images, label = val_generator.next_batch(batch_size)
            eval_sess = sess.run(fch, feed_dict={x: images})
            print(eval_sess)
            w_res = toBinaryString(eval_sess)
            wzy = toString(eval_sess)
            for j in range(batch_size):
                # print(wzy[j])
                file_res.write(wzy[j] + '\t' +  w_res[j] + '\t' + str(val_generator .images[k]) + '\n')
                k+=1
        k = 0
        for i in range(train_generator.dataSize // batch_size):
            images,label = train_generator.next_batch(batch_size)
            eval_sess = sess.run(fch, feed_dict={x: images})
            print(eval_sess)
            w_res = toBinaryString(eval_sess)
            wzy = toString(eval_sess)
            for j in range(batch_size):
                file_res.write(wzy[j] + '\t' + w_res[j] + '\t' + str(train_generator .images[k]) + '\n')
                k += 1

    file_res.close()
    sess.close()

if Train:
    train()
else:
    getHash()