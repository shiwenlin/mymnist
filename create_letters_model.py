import os
import tensorflow as tf
import pickle
import numpy as np
import argparse
from scipy.io import loadmat
import random

#load_mat_data translate mnist
def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.

        Arguments:
            mat_file_path: path to the .mat, should be in sample/

        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing

        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)

    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        a = np.array(img)
        c = np.reshape(a,(28,28))
        flipped = np.fliplr(img)

        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    #mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    #pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])


    # [:max_] mean all array
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, width,height,1)
    train_max = max_

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)

    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    test_max = max_


    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100))
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape training data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100))
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    training_images = training_images.reshape(train_max, width*height)
    testing_images = testing_images.reshape(test_max, width * height)


    datas = mat['dataset'][0][0][0][0][0][1]
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]


    array = np.zeros(datas.shape[0]*26,dtype=np.uint8)
    array = array.reshape((datas.shape[0],26))

    test_array = np.zeros(testing_labels.shape[0] * 26, dtype=np.uint8)
    test_array = test_array.reshape((testing_labels.shape[0], 26))

    for i in range(0,datas.shape[0]):
        result_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        value = datas[i].max()-1
        result_arr[value] = 1

        array[i] = np.array(result_arr)

    for i in range(0,testing_labels.shape[0]):
        result_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        value = testing_labels[i].max()-1
        result_arr[value] = 1

        test_array[i] = np.array(result_arr)


    training_labels = array
    testing_labels = test_array

    del datas
    del array
    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)







    # Convert type to float32
    testing_images = testing_images.astype('float32')
    training_images = training_images.astype('float32')


    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

#    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels))

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





def train_model (train_data):
    sess = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 26])
    W = tf.Variable(tf.zeros([784, 26]))
    b = tf.Variable(tf.zeros([26]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

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

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 26])
    b_fc2 = bias_variable([26])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Define loss and optimizer
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    """
    Train the model and save the model to disk as a model2.ckpt file
    file is stored in the same directory as this python script is started
    
    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    #with tf.Session() as sess:
        #sess.run(init_op)
    for i in range(30000):
       if i < 2495 :
           train_batch = training_data[0][0][i*50:i*50+50]
           lable_batch = training_data[0][1][i*50:i*50+50]
       else:
           tep = random.randint(0, training_data[0][0].shape[0] - 50)
           train_batch = training_data[0][0][tep:tep + 50]
           lable_batch = training_data[0][1][tep:tep + 50]
       if i%100 == 0:
         train_accuracy = accuracy.eval(feed_dict={
             x:train_batch, y_: lable_batch, keep_prob: 1.0})
         print("step %d, training accuracy %g"%(i, train_accuracy))
       train_step.run(feed_dict={x: train_batch, y_: lable_batch, keep_prob: 0.5})

    save_path = saver.save(sess, "./letter/model_letter.ckpt")
    print ("Model saved in file: ", save_path)


    print("test accuracy %g"%accuracy.eval(feed_dict={x: training_data[1][0][0:2000], y_: training_data[1][1][0:2000], keep_prob: 1.0}))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--file', type=str, help='Path .mat file data', default="./sample/emnist-letters")
    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--max', type=int, default=None, help='Max amount of data to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    print(args)

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(args.file, width=args.width, height=args.height, max_=args.max, verbose=args.verbose)



    train_model(training_data)
