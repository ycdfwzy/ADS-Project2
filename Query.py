import os
import cv2
import tensorflow as tf
import numpy as np
from DataGenerator import ImageDataGenerator
from DHN import *

batch_size = 100
hash_K = 48
checkpoint_path = os.path.join(os.getcwd(), "tmp\\dhn")

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


def get_hanming(binarystr1, binarystr2):
    L = len(binarystr1)
    dis = 0
    #print(len(binarystr1))
    #print(len(binarystr2))
    for i in range(L):
        if binarystr1[i] != binarystr2[i]:
            dis += 1
    return dis

def get_manhattan(originfeature1, originfeature2):
    if (len(originfeature1) != len(originfeature2)):
        print("Differnet Dimensions!")
        return
    dis = 0
    for i in range(len(originfeature1)):
        dis += abs(originfeature1[i]-originfeature2[i])
    return dis


def file_read(path):
    file_in = open(path, 'r')
    lines = file_in.read()
    ls = lines.split('\n')
    names = []
    features = []
    origin_features = []
    for line in ls:
        tmp = line.split('\t')
        if len(line) < 3: continue
        # print(tmp)
        temp = tmp[0].split(' ')
        ori = []
        for digstr in temp:
            if len(digstr) == 0: continue
            ori.append((float)(digstr))
        origin_features.append(ori)
        # print(origin_features[-1])
        # print(len(origin_features[-1]))
        features.append(tmp[1])
        names.append(tmp[2])
    return names, features, origin_features

def getSecond(x):
    return x[1]

def test_accuracy_dhn(names, features, originfeature):
    tot = 0
    less10 = 0
    for i in range(len(features)):
        cnt = 0
        acc = 0
        index_tmp = []
        print(i)
        for j in range(len(features)):
            #if (get_manhattan(originfeature[i], originfeature[j]) < 50.):
            index_tmp.append( (j, get_manhattan(originfeature[i], originfeature[j]), get_hanming(features[i], features[j])) )

        # print(len(index))

        index_tmp.sort(key=getSecond)
        index = []
        for j in range(len(index_tmp)):
            if j >= 100: break
            index.append( (index_tmp[j][0], index_tmp[j][2]) )
        index.sort(key=getSecond)

        for j in range(len(index)):
            if j >= 10: break
            cnt += 1
            # print(names[index[j][0]])
            if (names[index[j][0]].split('_')[0] == names[i].split('_')[0]):
                acc += 1
        if (cnt < 10):
            less10 += 1
        tot += (float)(acc/cnt)
        print((float)(acc/cnt))
    print(tot/len(features))
    print(less10)

def getHash(path):
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='x')
    model = DeepHashingNet(x, 1, hash_K, [])
    fch = model.fch
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver(tf.all_variables())

    sess = tf.InteractiveSession()

    query_generator = ImageDataGenerator(path,
                                         horizontal_flip=False, shuffle=False)

    file_res = open('query_dhn.txt', 'w')
    # sys.stdout = file_res
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(checkpoint_path, ckpt_name))
        print('Loading success, global_step is %s' % global_step)

        k = 0
        for i in range(query_generator.dataSize // batch_size):
            print(str(i) + "th batch in " + str(query_generator.dataSize // batch_size))
            images, label = query_generator.next_batch(batch_size)
            eval_sess = sess.run(fch, feed_dict={x: images})
            # print(eval_sess)
            w_res = toBinaryString(eval_sess)
            wzy = toString(eval_sess)

            for j in range(batch_size):
                file_res.write(wzy[j] + '\t' + w_res[j] + '\t' + str(query_generator .images[k]) + '\n')
                k+=1

    file_res.close()
    sess.close()

def showimage(name, showname = 'wzy'):
    if os.path.exists(os.path.join('./data/test', name)):
        img = cv2.imread(os.path.join('./data/test', name))
    elif os.path.exists(os.path.join('./data/train', name)):
        img = cv2.imread(os.path.join('./data/train', name))
    elif os.path.exists(os.path.join('./data/query', name)):
        img = cv2.imread(os.path.join('./data/query', name))
    cv2.imshow(showname, img)
    cv2.waitKey()

def query_dhn(names_dataset, features_dataset, originfeature_dataset, names_query, features_query, originfeature_query):
    file_out = open("answer_dhn.txt", "w")
    for i in range(2):
        index_tmp = []
        print(i)
        showimage(names_query[i], str(i))
        for j in range(len(features_dataset)):
            index_tmp.append( (j, get_manhattan(originfeature_query[i], originfeature_dataset[j]), get_hanming(features_query[i], features_dataset[j])) )

        index_tmp.sort(key=getSecond)
        index = []
        for j in range(len(index_tmp)):
            if j >= 100: break
            index.append( (index_tmp[j][0], index_tmp[j][2]) )
        index.sort(key=getSecond)

        file_out.write(names_query[i] + ':')
        index.sort(key=getSecond)
        for j in range(len(features_dataset)):
            if j >= 10: continue
            showimage(names_dataset[index[j][0]], str(i)+" "+str(j))
            file_out.write(" " + names_dataset[index[j][0]])
        file_out.write("\n")
    file_out.close()


names_dhn, features_dhn, origin_features  = file_read(r"C:\Users\ycdfw\Desktop\DHN_0.1\result_dhn.txt")
#getHash("data/query")
names_query_dhn, features_query_dhn, origin_features_query = file_read(r"C:\Users\ycdfw\Desktop\DHN_0.1\query_dhn.txt")
#test_accuracy_dhn(names_dhn, features_dhn, origin_features)
query_dhn(names_dhn, features_dhn, origin_features, names_query_dhn, features_query_dhn, origin_features_query)

