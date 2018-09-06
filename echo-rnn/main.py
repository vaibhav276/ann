from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)
# sess = tf.InteractiveSession()

num_epochs = 10
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# for next state
# input is state_size+1 -> 1 for current input and state_size for last state
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32, name='W')
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32, name='b')

# for output
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32, name='W2')
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32, name='b2')

# like transpose
# so each element in inputs_series will be of batch_size
# and there will be truncated_backprop_length such elements
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

optimizer = tf.train.AdagradOptimizer(0.3)
print ('tf version', tf.__version__)

# print(tf.shape(batchX_placeholder).eval())
# print(tf.shape(inputs_series).eval())

# x = tf.constant([[1,2,3], [4,5,6]])
# print(x.eval())
# y = tf.unstack(x, axis=1)
# print(y[0].eval()) # [1,4]
# print(y[1].eval()) # [2,5]
# print(y[2].eval()) # [3,6]

# print(tf.zeros([batch_size, state_size], tf.float32).eval())
# print(tf.zeros([batch_size, state_size], tf.float32))

# sess.close()

def generate_data():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)

def model_fit():
    current_state = init_state                # (batch_size, state_size)
    states_series = []
    for current_input in inputs_series:
        current_input = tf.reshape(current_input, [batch_size, 1])
        input_and_state_concatenated = tf.concat([
            current_input,                    # (batch_size, 1)
            current_state                     # (batch_size, state_size)
        ], 1)                                 # (batch_size, 1+state_size)

        next_state = tf.tanh(
            tf.matmul(
                input_and_state_concatenated, # (batch_size, 1+state_size)
                W                             # (state_size+1, state_size)
            ) + b                             # (1, state_size): automatically broadcasted
        )                                     # (batch_size, state_size)

        states_series.append(next_state)
        current_state = next_state

    # states_series                           # (batch_size, state_size) X truncated_backprop_length
    logits_series = map(
        (lambda s: tf.matmul(s, W2) + b2),
        states_series
    )                                         # (batch_size, num_classes) X truncated_backprop_length

    losses = map(
        (lambda (logits, labels):
         tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)),
        zip(logits_series, labels_series)
    )
    total_loss = tf.reduce_mean(losses)

    train_step = optimizer.minimize(total_loss)
    optimizer_initializer = tf.variables_initializer(optimizer.variables())

    return (logits_series, current_state, total_loss, train_step,
            optimizer_initializer)

def run_training():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        plt.ion()
        plt.figure()
        plt.show()
        loss_list = []

        for epoch_idx in range(num_epochs):
            x,y = generate_data()
            _init_state = np.zeros((batch_size, state_size))

            print('New data, epoch', epoch_idx)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:, start_idx:end_idx]
                batchY = y[:, start_idx:end_idx]

                (_logits_series, _current_state,
                _total_loss, _train_step, _opt_init) = sess.run(
                    model_fit(), feed_dict = {
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        init_state: _init_state
                    }
                )
                _init_state = _current_state

                _prediction_series = map(tf.nn.softmax, _logits_series)

                loss_list.append(_total_loss)

                if batch_idx % 100 == 0:
                    print('Step', batch_idx, 'Loss', _total_loss)
                    plot(loss_list, _prediction_series, batchX, batchY)

        plt.ioff()
        plt.show()


def plot(loss_list, prediction_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(prediction_series)[batch_series_idx]
        single_output_series = np.array(map(
            (lambda x: 1 if x[0] < 0.5 else 0),
            one_hot_output_series
        ))

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color='blue')
        plt.bar(left_offset, batchY[batch_series_idx, :]*0.5, width=1, color='red')
        plt.bar(left_offset, single_output_series*0.3, width=1, color='green')

    plt.draw()
    plt.pause(0.0001)

run_training()
