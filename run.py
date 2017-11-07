
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os
import re
import cv2
import random
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import math

#import helpers
import inference
import visualize
## Creating pairs
def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(15)]) - 1
    for d in range(15):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            labels.append(1)
            inc = random.randrange(1, 15)
            dn = (d + inc) % 15
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels.append(0)
    return np.array(pairs), np.array(labels)
##

##create batches
def next_batch(s,e,inputs,labels):
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    y= labels[s:e]
    return input1,input2,y
##
n_samples = 3016
batch_size = 128
# prepare data and tf.session
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
folder = "../input_data_resized"
listing1 = os.listdir(folder)
listing = sorted(listing1, key=lambda x: (int(re.sub('\D','',x)),x))
label=np.ones((n_samples,),dtype = 'int64')
label[0:316]=0
label[316:442]=1
label[442:754]=2
label[754:897]=3
label[897:1042]=4
label[1042:1340]=5
label[1340:1492]=6
label[1492:1702]=7
label[1702:2102]=8
label[2102:2279]=9
label[2279:2388]=10
label[2388:2559]=11
label[2559:2703]=12
label[2703:2872]=13
label[2872:]=14

names = ['pedestrian1','pedestrian2','pedestrian3','pedestrian4','pedestrian5','pedestrian6',
'pedestrian7','pedestrian8','pedestrian9','pedestrian10','pedestrian11','pedestrian12',
'pedestrian13','pedestrian14','pedestrian15']

images = []
for filename in listing:
    img = cv2.imread(os.path.join(folder,filename))
    images.append(img)

img_data = np.array(images)
img_data = img_data.astype('float32')
img_data /= 255
mu = img_data.mean()
img_data = img_data - mu
digit_indices = [np.where(label == i)[0] for i in range(15)]
x_pairs, y_pairs = create_pairs(img_data, digit_indices)
x_start,y_start = shuffle(x_pairs,y_pairs,random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x_start,y_start,test_size=0.2,random_state=4)

sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese();
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
new = True
model_ckpt = 'pedestrian.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training
if new:
    #saver.restore(sess, 'model.ckpt')
    total_batch = int(X_train.shape[0]/batch_size)
    for step in range(2001):
        total_correct = 0
        total_loss = 0
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            input1,input2,y =next_batch(s,e,X_train,y_train)
            _, loss_v, train_o1, train_o2, losses = sess.run([train_step, siamese.loss, siamese.o1, siamese.o2, siamese.debug], feed_dict={
                            siamese.x1: input1,
                            siamese.x2: input2,
                            siamese.y_: y})
            
            train_eucd2 = np.sum((train_o1-train_o2)**2,1,keepdims=True)
            train_eucd = np.sqrt(train_eucd2+1e-6)
            train_eucd = train_eucd.ravel()
            correct = np.argwhere(np.logical_or(np.logical_and(np.less(train_eucd,2.5),np.equal(y,1)),np.logical_and(np.greater(train_eucd,2.5),np.equal(y,0))))
            correct = correct.ravel()
            total_correct = total_correct+correct.size
            total_loss = total_loss+loss_v
        training_accuracy = 100*total_correct/(total_batch*batch_size)
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 1 == 0:
            print ('step %d: loss %.3f Training accuracy : %0.3f' % (step, total_loss,training_accuracy))
#        quit()
        if step % 10 == 0:
            saver.save(sess, 'pedestrian.ckpt')
            #embed = siamese.o1.eval({siamese.x1: X_test[0:128][0]})
            #embed.tofile('embed.txt')
else:
    saver.restore(sess, 'pedestrian.ckpt')

# visualize result
x_test = X_test.reshape([-1, 56, 112,3])
#visualize.visualize(em:wqbed, x_test)
