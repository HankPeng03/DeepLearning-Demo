import tensorflow as tf
import numpy as np


tf.reset_default_graph()

np.random.seed(0)
X = np.random.randn(2,4,5)

X[1,1:] = 0
seq_length = [4,1]
print('X:\n',X)

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=3,state_is_tuple=True)
gru = tf.nn.rnn_cell.GRUCell(3)

'''
  dynamic_rnn的用法
      input的大小[batch_size,sequence_max_length,embedding_size]
      若输入的example的长度不同，小于sequence_max_length的部分用0-padding方法填充
      sequence_length 指定每个example的长度，对于长度外的值不进行计算，其last_status重复上一步的last_status直至结束
  返回值：(outputs,last_status) 
        其中outputs是每一个迭代隐状态的输出(shape=[batch_size,sequence_max_length,embedding_size])
        last_status 由(c,h)组成(shape均为[batch_size,embedding_size])
         
'''

outputs, last_states = tf.nn.dynamic_rnn(cell,X,sequence_length=seq_length,dtype=tf.float64)
gruoutputs, grulast_states = tf.nn.dynamic_rnn(gru,X,sequence_length=seq_length,dtype=tf.float64)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

result,sta,gruout,grusta = sess.run([outputs,last_states,gruoutputs,grulast_states])

# gru没有状态输出，其状态就是最终输出，因为batch_size为2，所以输出为2
