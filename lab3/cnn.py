# import tensorflow as tf
# import numpy as np
# import os

# features = []; label = []
# filename = '.\\datasets\\original\\face'
# for train_class in os.listdir(filename):
#     for pic in os.listdir(filename+train_class):
#         features.append(filename+train_class+'/'+pic)
#         label.append(1)
# filename = '.\\datasets\\original\\nonface'
# for train_class in os.listdir(filename):
#     for pic in os.listdir(filename+train_class):
#         features.append(filename+train_class+'/'+pic)
#         label.append(0)
# temp = np.array([features,label])
# temp = temp.transpose()
# np.random.shuffle(temp)
# image_list = list(temp[:,0])
# label_list = list(temp[:,1])
# label_list = [tf.float32(i) for i in label_list]

# from PIL import Image
# face_path = '.\\datasets\\original\\face\\face_%03d.jpg'
# faces_path = []
# for i in range(500):
#     faces_path.append(face_path % i)
#
# nonface_path = '.\\datasets\\original\\nonface\\nonface_%03d.jpg'
# nonfaces_path = []
# for i in range(500):
#     nonfaces_path.append(nonface_path % i)
#
# imgs = []
# for i in range(500):
#     img = Image.open(faces_path[i])
#     img = img.convert('L').resize((24, 24))
#     imgs.append(img)
#
#     img = Image.open(nonfaces_path[i])
#     img = img.convert('L').resize((24, 24))
#     imgs.append(img)
#
# label = []
# for i in range(500):
#     if i%2 == 0:
#         label.append(1)
#     else:
#         label.append(-1)
#
# temp = np.array([imgs,label])

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------

file_dir = '.\\datasets\\original'

face = []
label_face = []
nonface = []
label_nonface = []

# step1：获取'E:/Re_train/image_data/training_image'下所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir):
    for file in os.listdir(file_dir + '\\face'):
        face.append(file_dir + '\\face' + '\\' + file)
        label_face.append(1)
    for file in os.listdir(file_dir + '\\nonface'):
        nonface.append(file_dir + '\\nonface' + '\\' + file)
        label_nonface.append(-1)

    # step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((face, nonface))
    label_list = np.hstack((label_face, label_nonface))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list, label_list

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    return all_image_list,all_label_list


# ---------------------------------------------------------------------------
# --------------------生成Batch----------------------------------------------

# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

# ========================================================================

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,2])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1]) #28*28

W_conv1 = weight_variable([5,5,1,32]) #channel=1
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool = max_pool_2x2(h_conv1) #14*14*32

W_conv2 = weight_variable([5,5,32,64]) #channel=32 -> 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #7*7*64

h_poo12_flat = tf.reshape(h_pool2,[-1,7*7*64])

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_poo12_flat,W_fc1)+b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

x,y = get_files(file_dir)
batch_xs = get_batch(x,y,100,100)[0]
batch_ys = get_batch(x,y,100,100)[1]

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(100):
    print(i)
    batch_xs = sess.run(batch_xs)
    batch_ys = sess.run(batch_ys)
    print(sess.run(batch_xs))
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(x,y))
