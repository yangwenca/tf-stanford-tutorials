"""
Simple logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import math
tf.set_random_seed(2)
# Define paramaters for the model
#learning_rate = 0.01
batch_size = 128
n_epochs = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('/Users/yangwen/Documents/Classes/cs20/tf-stanford-tutorials/data/mnist', one_hot=True) 

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# each lable is one hot vector.
X = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y_placeholder')
lr = tf.placeholder(tf.float32, name='Learningrate_placeholder')

pkeep = tf.placeholder(tf.float32, name='dropout')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y

# Method 1
#w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
#b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Adding layers
w1 = tf.Variable(tf.random_normal(shape=[5,5,1,4], stddev=0.1))
b1 = tf.Variable(tf.ones([4])/10)

w2 = tf.Variable(tf.random_normal(shape=[4,4,4,8], stddev=0.1))
b2 = tf.Variable(tf.ones([8])/10)

w3 = tf.Variable(tf.random_normal(shape=[4,4,8,12], stddev=0.1))
b3 = tf.Variable(tf.ones([12])/10)

w4 = tf.Variable(tf.random_normal(shape=[7*7*12, 200], stddev=0.1))
b4 = tf.Variable(tf.ones([200])/10)

w5 = tf.Variable(tf.random_normal(shape=[200, 10]))
b5 = tf.Variable(tf.ones([10])/10)

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
#Method 1
#logits = tf.matmul(X, w) + b 

## Adding layers
#y1 = tf.nn.relu(tf.matmul(X, w1)+b1)
#y1d = tf.nn.dropout(y1, pkeep)
#logits = tf.matmul(y1d, w2) + b2


y1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='same')+b1)
y2 = tf.nn.relu(tf.nn.covn2d(y1, w2, strides=[1,2,2,1], padding='same')+b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1,2,2,1], padding='same')+b3)
y3_reshape = tf.reshape(y3, shape=[-1,7*7*12])
y4 = tf.nn.relu(tf.matmul(y3_reshape, w4)+b4)
y4d = tf.nn.dropout(y4, pkeep)
logits = tf.matmul(y4d, w5)+b5


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
# batch_size is the key to make softmax_cross_entropy_with_logits to be equal as write explicitly
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))*batch_size
#loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

#loss = tf.reduce_mean(-tf.reduce_sum(Y *tf.log(tf.nn.softmax(logits))))

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./my_graph/03/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			max_learning_rate = 0.003
			min_learning_rate = 0.0001
			decay_speed = 2000.0
			dp = 0.75
			learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch, lr: learning_rate, pkeep: dp}) 
			total_loss += loss_batch
		print 'Average loss epoch {0}: {1}'.format(i, total_loss/n_batches)

	print 'Total time: {0} seconds'.format(time.time() - start_time)

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		learning_rate = 0.001
		dp = 1.0
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch, lr:learning_rate, pkeep: dp}) 
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)	
	
	print 'Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples)

	writer.close()
