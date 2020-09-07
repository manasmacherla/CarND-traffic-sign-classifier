# Load pickled data
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import random
import glob
import csv
import cv2
import os

EPOCHS = 20
BATCH_SIZE = 128

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
X_test = X_test.astype(np.float32)

image_shape = X_train[0].shape

def rgb2gray(rgb):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_shape = image_shape[0:2] + (1,)
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    gray = (gray - 128)/128
    gray = gray.reshape((-1,32,32,1))
    return gray.astype(np.float32)

def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    image_data = rgb2gray(image_data)
    normalized = (image_data)/255.0 - 0.5
    normalized = normalized.reshape((-1,32,32,1))
    return normalized.astype(np.float32)

def normalize_image(image_set):
    new_shape = image_shape[0:2] + (1,)
    new_vector = np.empty(shape=(len(image_set),) + new_shape, dtype=int)
    i = 0
    
    for image in image_set:
        #norm_img = cv2.normalize(image, np.zeros(image_shape[0:2]), 0, 255, cv2.NORM_MINMAX)
        norm_img = image/255.0 - 0.5
        norm_img = norm_img.astype(np.float32)
        gray_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        gray_img = np.reshape(gray_img, new_shape)
        np.append(new_vector,gray_img)    
    return new_vector

#image_set = rgb2gray(X_train)
#print(image_set)
  
X_train = rgb2gray(X_train)
X_valid = rgb2gray(X_valid)
X_test = rgb2gray(X_test)

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

"""
X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)
#X_train -= np.mean(X_train, axis = 0)

"""

print(X_train.shape, X_train.dtype)
print(X_valid.shape, X_valid.dtype)
print(X_test.shape, X_test.dtype)

"""
X_train -= np.mean(X_train)
X_train /= np.std(X_train, axis = 0)
X_valid -= np.mean(X_valid)
X_valid /= np.std(X_valid, axis = 0)
X_test -= np.mean(X_test)
X_test /= np.std(X_test, axis = 0)

"""

EPOCHS = 20
BATCH_SIZE = 128
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def LeNet(X):
    #     keep_prob = tf.placeholder(tf.float32)
    mu = 0
    sig = 0.1
    # Convoluted layer 1
    cn1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean=mu, stddev=sig))
    cn1_b = tf.Variable(tf.zeros(12))
    cn1 = tf.nn.conv2d(X, cn1_w, strides=[1, 1, 1, 1], padding='VALID') + cn1_b
    # Activation layer
    cn1 = tf.nn.relu(cn1)
    # Maxpool
    cn1 = tf.nn.max_pool(cn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    cn2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 25), mean=mu, stddev=sig))
    cn2_b = tf.Variable(tf.zeros(25))
    cn2 = tf.nn.conv2d(cn1, cn2_w, strides=[1, 1, 1, 1], padding='VALID') + cn2_b

    cn2 = tf.nn.relu(cn2)

    cn2 = tf.nn.max_pool(cn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    f0 = flatten(cn2)
    f0 = tf.nn.dropout(f0, keep_prob)
    dn1_w = tf.Variable(tf.truncated_normal(shape=(625, 300), mean=mu, stddev=sig))
    dn1_b = tf.Variable(tf.zeros(300))
    dn1 = tf.matmul(f0, dn1_w) + dn1_b

    dn1 = tf.nn.relu(dn1)

    dn2_w = tf.Variable(tf.truncated_normal(shape=(300, 100), mean=mu, stddev=sig))
    dn2_b = tf.Variable(tf.zeros(100))
    dn2 = tf.matmul(dn1, dn2_w) + dn2_b

    dn2 = tf.nn.relu(dn2)

    dn3_w = tf.Variable(tf.truncated_normal(shape=(100, 43), mean=mu, stddev=sig))
    dn3_b = tf.Variable(tf.zeros(43))
    dn3 = tf.matmul(dn2, dn3_w) + dn3_b

    logits = dn3

    return logits

"""

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    cnn1_w = tf.Variable(tf.truncated_normal(shape = (3,3,1,16), mean = mu, stddev = sigma))
    cnn1_b = tf.Variable(tf.zeros(16))
    cnn1 = tf.nn.conv2d(x, cnn1_w, strides = [1,1,1,1], padding = 'VALID') + cnn1_b
    
    # TODO: Activation.
    cnn1 = tf.nn.relu(cnn1)
    #cnn1 = tf.nn.dropout(cnn1, 0.8)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    cnn1 = tf.nn.max_pool(cnn1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    cnn2_w = tf.Variable(tf.truncated_normal(shape = (3,3,16,32), mean = mu, stddev = sigma))
    cnn2_b = tf.Variable(tf.zeros(32))
    cnn2 = tf.nn.conv2d(cnn1, cnn2_w, strides = [1,1,1,1], padding = 'VALID') + cnn2_b
    
    # TODO: Activation.
    cnn2 = tf.nn.relu(cnn2)
    #cnn2 = tf.nn.dropout(cnn2, 0.8)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    cnn2 = tf.nn.max_pool(cnn2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat1 = flatten(cnn2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fcn1_W = tf.Variable(tf.truncated_normal(shape=(6*6*32,500), mean = mu, stddev = sigma))
    fcn1_b = tf.Variable(tf.zeros(500))
    fcn1 = tf.add(tf.matmul(flat1, fcn1_W), fcn1_b)
    
    # TODO: Activation.
    fcn1 = tf.nn.relu(fcn1)    
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84
    fcn2_W = tf.Variable(tf.truncated_normal(shape=(500,500), mean = mu, stddev = sigma))
    fcn2_b = tf.Variable(tf.zeros(500))
    fcn2 = tf.add(tf.matmul(fcn1, fcn2_W), fcn2_b)
    
     # TODO: Activation.
    fcn2 = tf.nn.relu(fcn2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10
    fcn3_W = tf.Variable(tf.truncated_normal(shape=(500,43), mean = mu, stddev = sigma))
    fcn3_b = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fcn2, fcn3_W), fcn3_b)
                              
    return logits

"""

##################### TRAINING MODEL ###########################


x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='x')   # name your placeholder
y = tf.placeholder(tf.int32, (None), name='y')
one_hot_y = tf.one_hot(y, 43)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

################ EVALUATION #############################

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


#X_train, X_tst, y_train, y_tst = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")