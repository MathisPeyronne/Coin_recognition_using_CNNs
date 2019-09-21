"""
This code is the TensorFlow graph of the model.
It is needed in main.py to use the trained CNN.
"""
import tensorflow as tf
import pickle
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

################ Parameters ####################
NAME = 'lenet-V2'
EPOCHS = 10
BATCH_SIZE = 32
n_classes = 8
learning_rate = 0.0001

def LeNet(x):   
    """
    Function that builds the model in the TensorFlow graph.
    """
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases of each layer
    mu = 0
    sigma = 0.1    
    
    weights = {
        # The shape of the filter weight is (height, width, input_depth, output_depth)
        'conv1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)),
        'conv2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'fl1': tf.Variable(tf.truncated_normal(shape=(2704, 120), mean = mu, stddev = sigma)),
        'fl2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
        'out': tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    }

    biases = {
        # The shape of the filter bias is (output_depth,)
        'conv1': tf.Variable(tf.zeros(6)),
        'conv2': tf.Variable(tf.zeros(16)),
        'fl1': tf.Variable(tf.zeros(120)),
        'fl2': tf.Variable(tf.zeros(84)),
        'out': tf.Variable(tf.zeros(n_classes))
    }

    ################# Convolutional Layer 1 ################# 
    # Convolution. Input shape : (32, 32, 1). Output shape : (28, 28, 6).
    conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases['conv1'])
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input shape : (28, 28, 6). Output shape : (14, 14, 6).
    conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    
    ################# Convolutional Layer 2 ################# 
    # Convolution. Input shape: (14, 14, 6). Output shape : (10, 10, 16).
    conv2 = tf.nn.conv2d(conv1, weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases['conv2'])
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input shape : (10, 10, 16). Output shape : (5, 5, 16).
    conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # Flatten. Input shape : (5, 5, 16). Output shape : (400).
    fl0 = flatten(conv2)
    
    # Layer 3: Fully Connected Layer. Input shape : (400). Output shape : (120).
    fl1 = tf.add(tf.matmul(fl0, weights['fl1']), biases['fl1'])
    # Activation.
    fl1 = tf.nn.relu(fl1)
    
    # Layer 4: Fully Connected Layer. Input shape : (120). Output shape : (84).
    fl2 = tf.add(tf.matmul(fl1, weights['fl2']), biases['fl2'])
    # Activation.
    fl2 = tf.nn.relu(fl2)

    # Layer 5: Fully Connected Layer. Input shape : (84). Output shape : (10).
    logits = tf.add(tf.matmul(fl2, weights['out']), biases['out'])
                 
    return logits

############## Build the TensorFlow graph ##############

# Placeholders
x = tf.placeholder(tf.float32, (None, 64, 64, 1), name='x')
y = tf.placeholder(tf.int32, (None))
one_hot_y = y

# Specify the sequence of operations
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)
prediction_operation = tf.argmax(tf.nn.softmax(logits), 1) #cette ligne sera utilisée dans main.py

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Save the TensorFlow graph
saver = tf.train.Saver()  

def evaluate(X_data, y_data):
    """
    This function computes the accuracy of the current model. 
    X_data and y_data are test images that the network has not seen yet.
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    #Evaluate the test data one batch at the time.
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy / num_examples

def predict(X_data):
    """
    This function is used to get the prediction of the trained CNN.
    X_data are the images we want to classify.
    This function is used in main.py
    """
    sess = tf.get_default_session()
    # Run the prediction operation of the TensorFlow graph 
    # by giving it the data we want to predict.
    predictions = sess.run(prediction_operation, feed_dict={x: X_data})
    return predictions  
