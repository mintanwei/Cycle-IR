import tensorflow as tf
import numpy as np
from parameter_define import *
from scipy.misc import imread, imshow
from image_resizing import ResizeImage


# def normalizeNP(Vec):
#     shape = np.shape(Vec)[1]
#     VecMax = np.tile(np.max(Vec, 1, keepdims=True), [1, shape])
#     VecMin = np.tile(np.min(Vec, 1, keepdims=True), [1, shape])
#     return (Vec-VecMin)/(VecMax-VecMin)


def preprocess(featH, featW, input_size):
    newH = grid_size * featH
    newW = grid_size * featW

    B, M, N, C = input_size[0], input_size[1]/grid_size, input_size[2]/grid_size, input_size[3]
    input_size = tf.cast(input_size, 'float32')
    return newH, newW, B, M, N, input_size


def resize_images(images, aspect_ratio, input_size, B, M, N, newH, newW):
    img = tf.TensorArray(dtype=tf.float32, size=B, infer_shape=False)
    out_dims = [aspect_ratio[0], aspect_ratio[1], tf.shape(images)[-1]]

    init_state = (0, img)
    condition = lambda i, _: i < B

    def body(i, img):
        img_temp = images[i, ...]
        ta_final_result = resize_image(img_temp, aspect_ratio, input_size, M, N, newH, newW, i)
        ta_final_result = ResizeImage(ta_final_result, tf.constant([1., 0, 0, 0, 1., 0]), out_dims=out_dims)
        return [tf.add(i, 1), img.write(i, ta_final_result)]

    n, img_final = tf.while_loop(condition, body, init_state)
    return img_final.stack()


def resize_image(images, aspect_ratio, input_size, M, N, newH, newW, i):
    M = tf.cast(M, tf.int32)
    N = tf.cast(N, tf.int32)
    ta_final_result = tf.cond(tf.abs(aspect_ratio[0] - input_size[1]) > 1,
                              lambda: resize_columns(images, M, newH, i),
                              lambda: images)

    resized_images = tf.transpose(ta_final_result, [1, 0, 2])

    ta_final_result = tf.cond(tf.abs(aspect_ratio[1] - input_size[2]) > 1,
                              lambda: resize_columns(resized_images, N, newW, i),
                              lambda: resized_images)

    ta_final_result = tf.transpose(ta_final_result, [1, 0, 2])

    return ta_final_result


def resize_columns(images, M, newH, i):
    ta = tf.TensorArray(dtype=tf.float32, size=M, infer_shape=False)
    new_size = newH[i]

    theta = tf.constant([1., 0, 0, 0, 1., 0])

    init_state = (0, ta)
    condition = lambda j, _: j < M

    def body(j, ta):
        block = images[j * grid_size:(j + 1) * grid_size, :, :]
        target_shape = tf.cast(tf.shape(images), tf.float32)
        out_dims = [new_size[j], target_shape[1], target_shape[2]]

        new_block = tf.cond(new_size[j] >= 1,
                lambda: ResizeImage(block, theta, out_dims=out_dims),
                lambda: tf.zeros((0, tf.shape(images)[1], tf.shape(images)[2])))
        return [tf.add(j, 1), ta.write(j, new_block)]

    n, ta_final = tf.while_loop(condition, body, init_state)
    return ta_final.concat()


def reconstruct_image(images, featH, featW, input_size, aspect_ratio):
    newH, newW, B, M, N, input_size = preprocess(featH, featW, input_size)
    ta_final_result = resize_images(images, aspect_ratio, input_size, B, M, N, newH, newW)
    return ta_final_result


if __name__ == '__main__':
    img = imread('./RetargetMeAll/boat.png')
    img = np.stack((img, img, img, img, img, img))
    B, H, W, C = img.shape
    print(H/grid_size)
    print(W/grid_size)
    print(img.shape)

    images = tf.placeholder(tf.float32, shape=(B, H, W, C))
    featH = tf.Variable(initial_value=np.random.uniform(low=0.0, high=1.0, size=(B, np.int32(H/grid_size))), dtype=tf.float32)
    featW = tf.Variable(initial_value=np.random.uniform(low=0.0, high=1.0, size=(B, np.int32(W/grid_size))), dtype=tf.float32)
    input_size = tf.shape(images)
    aspect_ratio = tf.constant([np.round(H / 1.0), W / 3])

    # featH = tf.nn.sigmoid(featH)
    # featH = tf.clip_by_value(featH, 0, 1.0)
    # featW = tf.nn.sigmoid(featW)
    # featW = tf.clip_by_value(featW, 0, 1.0)

    ta_final_result = reconstruct_image(images, featH, featW, input_size, aspect_ratio)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = sess.run(ta_final_result, feed_dict={images: img})

        x = np.array(x)
        print(x.shape)
        imshow(x[0,...])
