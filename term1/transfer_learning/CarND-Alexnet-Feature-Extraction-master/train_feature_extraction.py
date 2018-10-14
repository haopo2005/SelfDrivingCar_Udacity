import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle
from time import time
from scipy.misc import imread
import pandas as pd
# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
    train = pickle.load(f)

with open('test.p', 'rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
nb_classes = len(np.unique(y_train))

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized =  tf.image.resize_images(x, [227, 227])
one_hot_y = tf.one_hot(y, nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
mu = 0
sigma = 0.1
jst_W  = tf.Variable(tf.truncated_normal(shape=shape, mean = mu, stddev = sigma))
jst_b  = tf.Variable(tf.zeros(nb_classes))
jst_out = tf.matmul(fc7, jst_W) + jst_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
##如果使用tf.nn.sparse_softmax_cross_entropy_with_logits，就不需要手动取转换one-hot了
'''
Labels used in softmax_cross_entropy_with_logits are the one hot version of labels used in sparse_softmax_cross_entropy_with_logits.
'''
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=jst_out)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#train_op = opt.minimize(loss_op, var_list=[jst_W, jst_b]),也许效率会更好
#var_list: Optional list or tuple of Variable objects to update to minimize loss
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(jst_out, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

EPOCHS = 5
BATCH_SIZE = 128

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

start = time()
# TODO: Train and evaluate the feature extraction model.

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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_val, y_val)
        training_accuracy = evaluate(X_train, y_train)
        print("EPOCH %d - %d sec ..."%(i+1, time() - start))
        print("Training Accuracy = {:.3f} Validation Accuracy = {:.3f}".format(training_accuracy, validation_accuracy))
        print()
        
    saver.save(sess, './alexnet')
    print("Model saved")


preds = tf.arg_max(jst_out, 1)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    # Read Images
    im1 = imread("construction.jpg").astype(np.float32)
    im2 = imread("stop.jpg").astype(np.float32)
    output = sess.run(preds, feed_dict={x: [im1, im2]})
    
    # Print Output
    sign_names = pd.read_csv('signnames.csv')
    for input_im_ind in range(output.shape[0]):
        print("%s" % (sign_names.ix[output[input_im_ind]]))
