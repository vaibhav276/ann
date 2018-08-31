import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
import numpy as np
from create_features import create_featureset_and_labels
train_x, train_y, test_x, test_y = create_featureset_and_labels('pos.txt', 'neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2

batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

layers = [
    # input layer
    {
        'weights': tf.Variable(
            tf.random_normal([len(train_x[0]), n_nodes_hl1])
        ),
        'biases': tf.Variable(
            tf.random_normal([n_nodes_hl1])
        ),
        'activation': tf.nn.relu
    },
    # hidden layers
    {
        'weights': tf.Variable(
            tf.random_normal([n_nodes_hl1, n_nodes_hl2])
        ),
        'biases': tf.Variable(
            tf.random_normal([n_nodes_hl1])
        ),
        'activation': tf.nn.relu
    },
    {
        'weights': tf.Variable(
            tf.random_normal([n_nodes_hl2, n_nodes_hl3])
        ),
        'biases': tf.Variable(
            tf.random_normal([n_nodes_hl1])
        ),
        'activation': tf.nn.relu
    },
    # output layer
    {
        'weights': tf.Variable(
            tf.random_normal([n_nodes_hl3, n_classes])
        ),
        'biases': tf.Variable(
            tf.random_normal([n_classes])
        ),
        'activation': None
    }
]

def nn_predict(data):
    result = data
    for layer in layers:
        activation = layer['activation']
        result = tf.add(
            tf.matmul(
                result,
                layer['weights']
            ),
            layer['biases']
        )
        if activation: result = activation(result)

    return result

def nn_train_and_evaluate(x):
    prediction = nn_predict(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=prediction,
            labels=y
        )
    )

    optimizer = tf.train.AdamOptimizer() # learning_rate = 0.001 (default)
    backprop = optimizer.minimize(cost)

    # epoch = forward + back prop
    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for epoch in range(n_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                i += batch_size

                # Evaluating cost forces forward propagation
                # And we explicitly run backprop
                # Variables are updates as side effect of backprop
                _, c = sess.run([backprop, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y
                                })
                epoch_loss += c
            print('Epoch: ' + str(epoch + 1)
                  + '/' + str(n_epochs)
                  + ' - loss = ' + str(epoch_loss))

        correct = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(y, 1)
        )

        accuracy = tf.reduce_mean(
            tf.cast(correct, 'float')
        )

        print('Accuracy: ', accuracy.eval({
            x: test_x,
            y: test_y
        }))

nn_train_and_evaluate(x)
