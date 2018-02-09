import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data/", one_hot=True)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biasis': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biasis': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biasis': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biasis': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biasis'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biasis'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biasis'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biasis']

    return output


sess = tf.Session()

File_Writer = tf.summary.FileWriter('graph', sess.graph)
print(sess.run(tf.global_variables_initializer()))
sess.close()
