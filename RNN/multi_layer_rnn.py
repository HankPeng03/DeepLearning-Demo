import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def multi_layer_static_lstm(input_x,n_steps,n_hidden):
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0))

    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    hiddens,states = tf.nn.static_rnn(cell=mcell,inputs=input_x1,dtype=tf.float32)
    return hiddens,states


def multi_layer_static_gru(input_x,n_steps,n_hidden):
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.nn.rnn_cell.GRUCell(num_units=n_hidden))

    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    hiddens,states = tf.nn.static_rnn(cell=mcell,inputs=input_x1,dtype=tf.float32)
    return hiddens,states

def multi_layer_static_mix(input_x,n_steps,n_hidden):
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)

    mcell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell,gru_cell])
    hiddens,states = tf.nn.static_rnn(cell=mcell,inputs=input_x1,dtype=tf.float32)
    return hiddens,states

def multi_layer_dynamic_lstm(input_x,n_steps,n_hidden):
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0))
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
    hiddens,states = tf.nn.dynamic_rnn(cell=mcell,inputs=input_x,dtype=tf.float32)

    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,states

def multi_layer_dynamic_gru(input_x,n_steps,n_hidden):
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.nn.rnn_cell.GRUCell(num_units=n_hidden))
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
    hiddens,states = tf.nn.dynamic_rnn(cell=mcell,inputs=input_x,dtype=tf.float32)

    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,states

def multi_layer_dynamic_mix(input_x,n_steps,n_hidden):
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    mcell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell,gru_cell])
    hiddens,states = tf.nn.dynamic_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)
    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,states

def mnist_rnn_classification(flag):

    tf.reset_default_graph()
    # 导入数据
    mnist = input_data.read_data_sets('MNIST-data',one_hot=True)

    n_input = 28
    n_steps = 28
    n_hidden = 128
    

