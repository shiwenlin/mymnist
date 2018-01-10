#import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist = None
model_path = "./model.ckpt"

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
#
# sess.run(tf.initialize_all_variables())
#
# y = tf.nn.softmax(tf.matmul(x,W) + b)
#
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#
#
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# for i in range(1000):
#   batch = mnist.train.next_batch(50)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_path = saver.save(sess, model_path)
print ("Model saved in file: ", save_path)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# sess.run(tf.initialize_all_variables())
# for i in range(20000):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print "step %d, training accuracy %g"%(i, train_accuracy)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# """
# print "test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
# """
#
# #save result
# saver = tf.train.Saver()
# saver.save(sess,model_path)
#
# for i in xrange(10):
#     testSet = mnist.test.next_batch(50)
#     print("test accuracy %g"%accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))