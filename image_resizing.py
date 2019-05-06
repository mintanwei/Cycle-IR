import tensorflow as tf
import numpy as np
from scipy.misc import imread, imshow


def ResizeImage(input_fmap, theta, out_dims=None, **kwargs):
    theta = tf.reshape(theta, [2, 3])
    H = out_dims[0]
    W = out_dims[1]
    batch_grids = affine_grid_generator(H, W, theta)
    x_s = tf.squeeze(batch_grids[0:1, :, :], 0)
    y_s = tf.squeeze(batch_grids[1:2, :, :], 0)
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    return out_fmap


def get_pixel_value(img, x, y):
    indices = tf.stack([y, x], 2)
    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    x = tf.range(0, width) * tf.divide(1.0, width)
    y = tf.range(0, height) * tf.divide(1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to (x_t, y_t , 1)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')
    batch_grids = tf.matmul(theta, sampling_grid)
    batch_grids = tf.reshape(batch_grids, [2, tf.shape(y)[0], tf.shape(x)[0]])
    return batch_grids


def bilinear_sampler(img, x, y):
    H = tf.shape(img)[0]
    W = tf.shape(img)[1]
    C = tf.shape(img)[2]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W/H]
    x = x * tf.cast(W, 'float32')
    y = y * tf.cast(H, 'float32')

    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(x - tf.mod(x, tf.floor(x)), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(y - tf.mod(y, tf.floor(y)), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)  # x0, y0
    Ib = get_pixel_value(img, x0, y1)  # x0, y0+1
    Ic = get_pixel_value(img, x1, y0)  # x0, y0-1
    Id = get_pixel_value(img, x1, y1)  # x0+1, y0

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)
    wc = tf.expand_dims(wc, axis=2)
    wd = tf.expand_dims(wd, axis=2)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    weight = tf.add_n([wa, wb, wc, wd])
    out_final = tf.cast(weight > 0.001, tf.float32) * out + tf.cast(weight <= 0.001, tf.float32) * Ia

    return out_final


if __name__ == '__main__':
    img = imread('./RetargetMeAll/boat.png')
    H, W, C = img.shape
    print(img.shape)

    images = tf.placeholder(tf.float32, shape=(H, W, C))
    aspect_ratio = tf.placeholder(tf.float32, shape=(3,))
    theta = tf.Variable(tf.constant([1.0,0,0,0,1.0,0]), dtype=tf.float32)

    with tf.Session() as sess:
        ta_final_result = ResizeImage(images, theta, out_dims=aspect_ratio)
        sess.run(tf.global_variables_initializer())

        # Runs the op.
        x = sess.run(ta_final_result, feed_dict={images: img, aspect_ratio: [H, W * 0.5, C]})

        x = np.array(x)
        imshow(x)
        print(x.shape)
