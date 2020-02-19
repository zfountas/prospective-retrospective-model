################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
################################################################################

from numpy import *
import os
import numpy as np
import time
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf
from alexnet.classes import class_names

class AlexNet:

    def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


    def __init__(self, first_layer='input', last_layer='output', spatial_attention=False, params=dict()):
        self.name = 'AlexNet(' + first_layer + '-' + last_layer + ')'
        train_x = zeros((1, 227,227,3)).astype(float32)
        train_y = zeros((1, 1000))
        xdim = train_x.shape[1:]
        ydim = train_y.shape[1]
        print("Creating TF network with output layer:",last_layer)
        self.best_label_prob = [] # keeps the last '30' probabilities
        self.best_label_prob_window = 30

        if last_layer == 'output':
            self.end_of_network = True
            self.best_label = 'Nothing yet'
        else:
            self.end_of_network = False
            self.best_label = '-'

        # This is the current predicted prior: Mn,k
        self.generative_model_prediction = np.array([])
        self.spatial_attention = spatial_attention
        ########################################################################

        #In Python 3.5, change this to:
        net_data = load(open("alexnet/bvlc_alexnet.npy", "rb"), allow_pickle=True, encoding="latin1").item()
        #net_data = load("bvlc_alexnet.npy").item()
        prob_shape = (1, 1000)
        fc7_shape = (1, 4096)
        fc6_shape = (1, 4096)
        conv5_shape = (1, 13, 13, 256)
        conv4_shape = (1, 13, 13, 384)
        conv3_shape = (1, 13, 13, 384)
        conv2_shape = (1, 28, 28, 256)
        conv1_shape = (1, 57, 57, 96)


        self.stop_     = False
        self.started   = False

        if first_layer == 'input':
            self.started = True
            self.x = tf.placeholder(tf.float32, (None,) + xdim)
            self.in_ = self.x

            #conv1
            #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
            conv1W = tf.Variable(net_data["conv1"][0])
            conv1b = tf.Variable(net_data["conv1"][1])
            conv1_in = self.conv(self.x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w,
                                 padding="SAME", group=1)
            self.conv1 = tf.nn.relu(conv1_in)
            if last_layer == 'conv1':
                self.stop_ = True
                self.out = self.conv1

        if ((not self.stop_) and self.started) or first_layer == 'conv1':
            if first_layer == 'conv1':
                self.started = True
                self.conv1 = tf.placeholder(tf.float32, conv1_shape)
                self.in_ = self.conv1
        #self.conv1_ph = tf.placeholder(tf.float32, tf.shape(self.conv1))

            #lrn1
            #lrn(2, 2e-05, 0.75, name='norm1')
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(self.conv1,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias)

            #maxpool1
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1],
                                      strides=[1, s_h, s_w, 1], padding=padding)

            #conv2
            #conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
            conv2W = tf.Variable(net_data["conv2"][0])
            conv2b = tf.Variable(net_data["conv2"][1])
            conv2_in = self.conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w,
                                 padding="SAME", group=group)
            self.conv2 = tf.nn.relu(conv2_in)
            if last_layer == 'conv2':
                self.stop_ = True
                self.out = self.conv2

        if ((not self.stop_) and self.started) or first_layer == 'conv2':
            if first_layer == 'conv2':
                self.started = True
                self.conv2 = tf.placeholder(tf.float32, conv2_shape)
                self.in_ = self.conv2
            #lrn2
            #lrn(2, 2e-05, 0.75, name='norm2')
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(self.conv2, depth_radius=radius,
                                                             alpha=alpha,
                                                             beta=beta,
                                                             bias=bias)

            #maxpool2
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1],
                                      strides=[1, s_h, s_w, 1], padding=padding)

            #conv3
            #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
            conv3W = tf.Variable(net_data["conv3"][0])
            conv3b = tf.Variable(net_data["conv3"][1])
            conv3_in = self.conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w,
                                 padding="SAME", group=group)
            self.conv3 = tf.nn.relu(conv3_in)
            if last_layer == 'conv3':
                self.stop_ = True
                self.out = self.conv3

        if ((not self.stop_) and self.started) or first_layer == 'conv3':
            if first_layer == 'conv3':
                self.started = True
                self.conv3 = tf.placeholder(tf.float32, conv3_shape)
                self.in_ = self.conv3
            #conv4
            #conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
            conv4W = tf.Variable(net_data["conv4"][0])
            conv4b = tf.Variable(net_data["conv4"][1])
            conv4_in = self.conv(self.conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w,
                                 padding="SAME", group=group)
            self.conv4 = tf.nn.relu(conv4_in)
            if last_layer == 'conv4':
                self.stop_ = True
                self.out = self.conv4

        if ((not self.stop_) and self.started) or first_layer == 'conv4':
            if first_layer == 'conv4':
                self.started = True
                self.conv4 = tf.placeholder(tf.float32, conv4_shape)
                self.in_ = self.conv4
            #conv5
            #conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
            conv5W = tf.Variable(net_data["conv5"][0])
            conv5b = tf.Variable(net_data["conv5"][1])
            conv5_in = self.conv(self.conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w,
                                 padding="SAME", group=group)
            self.conv5 = tf.nn.relu(conv5_in)
            if last_layer == 'conv5':
                self.stop_ = True
                self.out = self.conv5

        if ((not self.stop_) and self.started) or first_layer == 'conv5':
            if first_layer == 'conv5':
                self.started = True
                self.conv5 = tf.placeholder(tf.float32, conv5_shape)
                self.in_ = self.conv5
            #maxpool5
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, k_h, k_w, 1],
                                      strides=[1, s_h, s_w, 1], padding=padding)

            #fc6
            #fc(4096, name='fc6')
            fc6W = tf.Variable(net_data["fc6"][0])
            fc6b = tf.Variable(net_data["fc6"][1])
            self.fc6 = tf.nn.relu_layer(tf.reshape(maxpool5,
                             [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
            if last_layer == 'fc6':
                self.stop_ = True
                self.out = self.fc6

        if ((not self.stop_) and self.started) or first_layer == 'fc6':
            if first_layer == 'fc6':
                self.started = True
                self.fc6 = tf.placeholder(tf.float32, fc6_shape)
                self.in_ = self.fc6
            #fc7
            #fc(4096, name='fc7')
            fc7W = tf.Variable(net_data["fc7"][0])
            fc7b = tf.Variable(net_data["fc7"][1])
            self.fc7 = tf.nn.relu_layer(self.fc6, fc7W, fc7b)
            if last_layer == 'fc7':
                self.stop_ = True
                self.out = self.fc7

        if ((not self.stop_) and self.started) or first_layer == 'fc7':
            if first_layer == 'fc7':
                self.started = True
                self.fc7 = tf.placeholder(tf.float32, fc7_shape)
                self.in_ = self.fc7
            #fc8
            #fc(1000, relu=False, name='fc8')
            fc8W = tf.Variable(net_data["fc8"][0])
            fc8b = tf.Variable(net_data["fc8"][1])
            fc8 = tf.nn.xw_plus_b(self.fc7, fc8W, fc8b)

            #prob
            #softmax(name='prob'))
            self.prob = tf.nn.softmax(fc8)
            self.out = self.prob

        # define a placeholder that will hold our target value
        self.christos = tf.placeholder(tf.float32, self.out.get_shape())

        # The loss function for softmax should be cross-entropy and for
        # linear layer should be mean squared error
        self.loss = tf.reduce_sum( tf.square(self.out - self.christos) )

        # learning rate 0.0001
        self.optimize = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

        # Define the graph that runs backpropagation
        self.input_grads = tf.gradients(self.loss, self.in_)

        # Remove the first dimension because it's not yet set and represents batch size
        self.gradients = np.zeros(tuple(list(self.in_.get_shape())[1:]))
        self.grad_corrector = params["mask"]["grad_corrector"]
        self.grad_error = 0.0
        self.mask_baseline = params["mask"]["baseline"]
        self.mask_average = params["mask"]["average"]
        self.change_rate = params["mask"]["change_rate"]

        # Used to save the complete raw version of the mask (i.e. unnormalized)
        # We save it in case it is needed for logs etc by the file that calls
        # sensory_system2
        self.raw_mask = []

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    # Keep save_raw_mask false if no logging is required and we need to optimize
    # speed/memory consumption..
    def update(self, frame, save_raw_mask=False):
        start_time = time.time()

        # NOTE: output.shape[0] will be 1 because we provide only one image!
        if self.end_of_network:
            inds = argsort(self.activations)[0,:]
            self.best_label = class_names[inds[-1]] + ": " + str(self.activations[0,inds[-1]])
            self.best_label_prob.append(float(self.activations[0,inds[-1]]))
            # Make sure that we keep only the last 'best_label_prob_window' probs..
            while len(self.best_label_prob) > self.best_label_prob_window:
                self.best_label_prob = self.best_label_prob[1:]

        self.time_took = time.time() - start_time


def __main__():
    alexnet = AlexNet()
    alexnet.update(frame)
