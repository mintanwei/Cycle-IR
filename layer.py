import tensorflow as tf
import numpy as np
from spatial_transformer_network import spatial_transformer_network


tf.set_random_seed(1000)

def conv2d_relu(layer_name, input, kernel_shape, strides_shape, padding_type='SAME'):
    with tf.variable_scope(layer_name):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # Create variable named "biases".
        biases = tf.get_variable("biases", [kernel_shape[-1]], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, filter=weights,
                            strides=strides_shape, padding=padding_type, name='covolution')
        return tf.nn.relu(conv + biases, name='relu')


def max_pool2d(layer_name, input, kernel_size, strides_shape):
    with tf.variable_scope(layer_name):
        return tf.nn.max_pool(input, ksize=kernel_size, strides=strides_shape, padding='SAME', name='max_pooling')


def mean_pool2d(layer_name, input, kernel_size, strides_shape):
    with tf.variable_scope(layer_name):
        return tf.nn.avg_pool(input, ksize=kernel_size, strides=strides_shape, padding='SAME', name='mean_pooling')


def fc2d_relu(layer_name, input, output_dim):
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights', [input.shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable('biases', [output_dim],
                            initializer=tf.constant_initializer(0.1))
        return tf.nn.relu(tf.matmul(input, w) + b, name='relu')


def fc2d_sigmoid(layer_name, input, output_dim):
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights', [input.shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable('biases', [output_dim],
                            initializer=tf.constant_initializer(0.1))
        return tf.nn.sigmoid(tf.matmul(input, w) + b, name='sigmoid')


def dropout2d(layer_name, input, keep_prob):
    with tf.variable_scope(layer_name):
        return tf.nn.dropout(input, keep_prob, name='dropout')


def softmax2d(layer_name, input, num_class):
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights', [input.shape[-1], num_class],
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable('biases', [num_class],
                            initializer=tf.random_normal_initializer(0.1))
        logits = tf.matmul(input, w) + b
        return logits


def BatchNormalization(input_tensor, is_training, use_relu=False, layer_name=None):
    """
    Handy wrapper function for convolutional networks.

    Performs batch normalization on the input tensor.
    """
    with tf.variable_scope(layer_name):
        normed = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=is_training)

        if use_relu:
            normed = tf.nn.relu(normed)

        return normed


def SENet(layer_name, input):
    with tf.variable_scope(layer_name):
        x_conv = conv2d_relu('conv', input, kernel_shape=[3, 3, input.shape[-1], input.shape[-1] * 2],
                             strides_shape=[1, 1, 1, 1], padding_type='SAME')
        x = tf.reduce_mean(x_conv, [1, 2])
        x_flat, num_feature = flatten(x)
        x = softmax2d('fc1', x_flat, num_feature/16)
        x_fc2 = fc2d_sigmoid('fc2_sigmoid', x, input.shape[-1]*2)
        x_fc2 = tf.reshape(x_fc2, [tf.cast(x_fc2.shape[0], tf.int32), 1, 1, tf.cast(x_fc2.shape[-1], tf.int32)])
        x = x_conv * x_fc2
        return x


def flatten(input):
    """
    Handy function for flattening the result of a conv2D or
    maxpool2D to be used for a fully-connected (affine) layer.
    """
    layer_shape = input.get_shape()
    # num_features = tf.reduce_prod(tf.shape(layer)[1:])
    input_size = layer_shape[1:].num_elements()
    input_flat = tf.reshape(input, [-1, input_size])
    return input_flat, input_size


def inception_v3(layer_name, x, in_channels, out_1x1, out_rd3x3, out_3x3, out_rd5x5, out_5x5, out_poolproj):
    with tf.variable_scope(layer_name):
        # 1x1
        filter_1x1 = tf.get_variable('1x1_weights', [1, 1, in_channels, out_1x1],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_1x1 = tf.nn.conv2d(x, filter_1x1, strides=[1, 1, 1, 1], padding='SAME')

        # 3x3 reduce
        filter_reduce_3x3 = tf.get_variable('3x3_reduce_weights', [1, 1, in_channels, out_rd3x3],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_reduce_3x3 = tf.nn.conv2d(x, filter_reduce_3x3, strides=[1, 1, 1, 1], padding='SAME')
        # 3x3
        filter_3x3 = tf.get_variable('3x3_weights', [3, 3, out_rd3x3, out_3x3],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_3x3 = tf.nn.conv2d(conv_reduce_3x3, filter_3x3, strides=[1, 1, 1, 1], padding='SAME')

        # 5x5 reduce
        filter_reduce_3x3 = tf.get_variable('5x5_reduce_weights', [1, 1, in_channels, out_rd5x5],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_reduce_5x5 = tf.nn.conv2d(x, filter_reduce_3x3, strides=[1, 1, 1, 1], padding='SAME')
        # 5x5 v1
        filter_v1_5x5 = tf.get_variable('5x5_v1_weights', [3, 3, out_rd5x5, out_5x5],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_v1_5x5 = tf.nn.conv2d(conv_reduce_5x5, filter_v1_5x5, strides=[1, 1, 1, 1], padding='SAME')
        # 5x5 v2
        filter_v2_5x5 = tf.get_variable('5x5_v2_weights', [3, 3, out_5x5, out_5x5],
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_v2_5x5 = tf.nn.conv2d(conv_v1_5x5, filter_v2_5x5, strides=[1, 1, 1, 1], padding='SAME')

        # avg pooling
        pooling = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # 1x1
        filter_pool_1x1 = tf.get_variable('1x1_pool_weights', [1, 1, in_channels, out_poolproj],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        conv_pool = tf.nn.conv2d(pooling, filter_pool_1x1, strides=[1, 1, 1, 1], padding='SAME')

        # bias dimension = 3*filter_count and then the extra in_channels for the avg pooling
        bias = tf.get_variable('bias', shape=[out_1x1 + out_3x3 + out_5x5 + out_poolproj],
                               initializer=tf.random_normal_initializer(0.1))
        x = tf.concat([conv_1x1, conv_3x3, conv_v2_5x5, conv_pool], axis=3)  # Concat in the 4th dim to stack
        x = tf.nn.bias_add(x, bias)
        return tf.nn.relu(x)


def residual_net(layer_name, input):
    with tf.variable_scope(layer_name):
        x1 = conv2d_relu('conv1', input, kernel_shape=[3, 3, input.shape[-1], input.shape[-1]*2], strides_shape=[1, 1, 1, 1],
                         padding_type='SAME')
        x2 = conv2d_relu('conv2', x1, kernel_shape=[3, 3, input.shape[-1]*2, input.shape[-1]*2], strides_shape=[1, 1, 1, 1],
                         padding_type='SAME')
        return x1 + x2


def parameter_localisation(layer_name, x):
    with tf.variable_scope(layer_name):
        x = residual_net('res1', x)
        x = max_pool2d('maxpool1', x, kernel_size=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

        x = residual_net('res2', x)
        x = max_pool2d('maxpool2', x, kernel_size=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

        x = residual_net('res3', x)
        x = max_pool2d('maxpool3', x, kernel_size=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

        x = residual_net('res4', x)
        x = max_pool2d('maxpool4', x, kernel_size=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

        x = SENet('SENet1', x)
        x = max_pool2d('maxpool5', x, kernel_size=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

        x = tf.reduce_mean(x, [1, 2])
        x_flat, num_feature = flatten(x)

        # identity transform
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()

        W_fc = tf.Variable(tf.zeros([num_feature, 6]), name='W_fc1')
        b_fc = tf.Variable(initial_value=initial, name='b_fc1')
        theta = tf.matmul(x_flat, W_fc) + b_fc
        return theta


def STN(layer_name, x, img_size):
    theta = parameter_localisation(layer_name, x)
    STN_Img = spatial_transformer_network(x, theta, img_size)
    return STN_Img
