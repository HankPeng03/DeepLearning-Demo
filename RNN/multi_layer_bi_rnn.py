import tensorflow as tf
import numpy as np

def single_layer_static_bi_lstm(input_x,n_steps,n_hidden):
    input_x1 = tf.unstack(value=input_x,num=n_steps,axis=1)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    # 返回的hiddens是list，每个元素是前向输出与后向输出的合并
    hiddens,fw_state,bw_state = tf.nn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                               cell_bw=lstm_bw_cell,inputs=input_x1,dtype=tf.float32)
    return hiddens,fw_state,bw_state

def singele_layer_dynamic_bi_lstm(input_x,n_steps,n_hidden):
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    hiddens,state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                    cell_bw=lstm_bw_cell,inputs=input_x,dtype=tf.float32)
    hiddens = tf.concat(hiddens,axis=2)
    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,state

def multi_layer_dynamic_bi_lstm(input_x,n_steps,n_hidden):

    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        stacked_fw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0))
        stacked_bw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0))
    hiddens,fw_state,bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw=stacked_fw_rnn,
                                                                cell_bw=stacked_bw_rnn,inputs=input_x,dtype=tf.float32)
    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,fw_state,bw_state

def multi_layer_static_bi_lstm(input_x,n_steps,n_hideen):
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    stacked_fw_rnn , stacked_bw_rnn = [], []
    for i in range(3):
        stacked_fw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hideen,forget_bias=1.0))
        stacked_bw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hideen, forget_bias=1.0))
    hiddens,fw_state,bw_state = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw=stacked_fw_rnn,
                                                   cell_bw=stacked_bw_rnn,inputs=input_x1,dtype=tf.float32)
    return hiddens,fw_state,bw_state

def mnist_rnn_classification(flag):
    pass



