import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def single_layer_static_lstm(input_x, n_steps, n_hidden):
    '''

    :param input_x:  输入张量，shape=[batch_size,n_steps,n_input]
    :param n_steps: 时序总数
    :param n_hidden: LSTM单元输出的节点个数，即隐藏层节点个数
    :return: 每个迭代的输出，与最新记忆状态
    '''
    # 将input_x按列拆分，返回一个有n_steps个张量组成的list
    # 若调用的是静态rnn函数，需要进行这样处理，将序列作为第一维度
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    # 静态rnn函数传入的是一个张量list，每一个元素都是一个(batch_size,n_input)大小的张量
    hiddens, states = tf.nn.static_rnn(cell=lstm_cell,inputs=input_x1, dtype=tf.float32)
    return hiddens, states


def single_layer_static_gru(input_x,n_steps,n_hidden):

    '''
      返回静态单层GRU单元的输出，以及cell状态
    :param input_x:
    :param n_steps:
    :param n_hidden:
    :return:
    '''

    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
    hiddens, states = tf.nn.static_rnn(cell=gru_cell,inputs=input_x1,dtype=tf.float32)
    return hiddens,states


def single_layer_dynamic_lstm(input_x,n_steps,n_hidden):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input] ，输出也是如此形状
    hiddens,states = tf.nn.dynamic_rnn(cell=lstm_cell,inputs=input_x,dtype=tf.float32)
    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,states

def single_layer_dynamic_gru(input_x,n_steps, n_hidden):
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
    hiddens,states = tf.nn.dynamic_rnn(cell=gru_cell,inputs=input_x,dtype=tf.float32)
    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,states

def mnist_rnn_classification(flag):
    '''
      对MNIST进行分类
    :param flag: 1-单层静态LSTM
                 2-单层静态GRU
                 3-单层动态LSTM
                 4-单层动态GRU
    :return:
    '''

    # 导入数据
    tf.reset_default_graph()
    print('开始导入数据')
    mnist = input_data.read_data_sets('MNIST-data',one_hot=True)
    print('导入数据成功')
    # 定义参数与网络结构
    n_input = 28   #输入节点数
    n_steps = 28   #序列长度
    n_hidden = 128   #隐藏层节点数
    n_classes = 10      #类别数
    batch_size = 128        #小批量大小
    training_step = 5000      #迭代次数
    display_step = 200
    learning_rate = 1e-4

    input_x = tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_input])
    input_y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

    if flag == 1:
        print('单层静态LSTM')
        hiddens,states = single_layer_static_lstm(input_x,n_steps,n_hidden)
    elif flag == 2:
        print('单层静态GRU')
        hiddens,states = single_layer_static_gru(input_x,n_steps,n_hidden)
    elif flag == 3:
        print('单层动态LSTM')
        hiddens,states = single_layer_dynamic_lstm(input_x,n_steps,n_hidden)
    elif flag == 4:
        print('单层动态GRU')
        hiddens,states = single_layer_dynamic_gru(input_x,n_steps,n_hidden)

    # 取最后一个时序的输出，经过全连接网络得到输出
    output = tf.layers.dense(inputs=hiddens[-1],units=n_classes,activation=tf.nn.softmax)
    # 定义代价函数
    cost = tf.reduce_mean(-tf.reduce_sum(input_y*tf.log(output),axis=1))
    # 定义优化
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # 预测结果评估
    correct = tf.equal(tf.argmax(output,1),tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

    # 创建list保存每一次迭代的结果
    test_accuracy_list = []
    test_cost_list = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(training_step):
            x_batch, y_batch = mnist.train.next_batch(batch_size=batch_size)
            x_batch = x_batch.reshape([-1,n_steps,n_input])

            sess.run([train],feed_dict={input_x:x_batch,input_y:y_batch})
            if (i+1)%display_step == 0:
                training_accuracy,training_cost = sess.run([accuracy,cost],
                                                           feed_dict={input_x:x_batch,input_y:y_batch})
                print('Step {0}:Training set accuracy {1} , cost {2}'.format(i+1,training_accuracy,training_cost))

        # 进行测试，分200次，一次测试50个样本
        for i in range(200):
            x_batch,y_batch = mnist.test.next_batch(batch_size=50)
            x_batch = x_batch.reshape([-1,n_steps,n_input])
            test_accuracy, test_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch,input_y:y_batch})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost)

            if (i+1)%20 == 0:
                print('Step {0}: Test set accuracy {1} cost {2}'.format(i+1, test_accuracy, test_cost))

        print('Test accuracy:{0}'.format(np.mean(test_accuracy_list)))

if __name__=='__main__':
    mnist_rnn_classification(1)
    mnist_rnn_classification(2)
    mnist_rnn_classification(3)
    mnist_rnn_classification(4)