import tensorflow as tf
import numpy as np
from parameter_define import *
from scipy.misc import imresize, imread, imshow, imsave
from image_resize_float import resize_image_float


def block_transform(newH, grid_size):
    greater_mask = tf.greater(newH, grid_size)
    less_mask = tf.less(newH, grid_size)
    addValue = tf.multiply(newH, tf.cast(greater_mask, tf.float32)) - tf.cast(greater_mask, tf.float32)*grid_size
    minValue = tf.reduce_sum(addValue, 1, True) / tf.reduce_sum(tf.cast(less_mask, tf.float32), 1, True)

    newH_less = newH - tf.tile(minValue, [1, tf.shape(newH)[1]])
    return tf.multiply(newH_less, tf.cast(less_mask, tf.float32)) + tf.cast(greater_mask, tf.float32)*grid_size


def preprocess(featH, featW, input_size, aspect_ratio):
    # normalize
    newH = grid_size * featH
    newW = grid_size * featW

    B, M, N, C = input_size[0], input_size[1]/grid_size, input_size[2]/grid_size, input_size[3]
    start = tf.zeros([B, 1], 'float32')
    input_size = tf.cast(input_size, 'float32')

    # newH = aspect_ratio[0] * featH / tf.tile(tf.reduce_sum(featH, 1, True), [1, tf.shape(featH)[1]])
    # newW = aspect_ratio[1] * featW / tf.tile(tf.reduce_sum(featW, 1, True), [1, tf.shape(featW)[1]])

    newH = tf.cumsum(newH, 1)
    newW = tf.cumsum(newW, 1)

    # with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
    #     newH = tf.round(newH)
    #     newW = tf.round(newW)

    # newH_ = tf.concat((newH[:,0:-1], newH[:,-2:-1] + (aspect_ratio[0] - newH[:,-2:-1])), 1)
    # newW_ = tf.concat((newW[:, 0:-1], newW[:, -2:-1] + (aspect_ratio[1] - newW[:, -2:-1])), 1)

    newH_ = tf.concat((start, newH), 1)
    newW_ = tf.concat((start, newW), 1)
    return newH_, newW_, B, M, N, input_size


def resize_images(images, aspect_ratio, input_size, B, M, N, newH, newW):
    img = tf.TensorArray(dtype=tf.float32, size=B, infer_shape=False)
    img2 = tf.TensorArray(dtype=tf.float32, size=B, infer_shape=False)
    out_dims = [aspect_ratio[0], aspect_ratio[1], tf.shape(images)[-1]]
    out_dims2 = [input_size[1], input_size[2], input_size[3]]

    init_state = (0, img, img2)
    condition = lambda i, _, _img: i < B

    def body(i, img, img2):
        img_temp = images[i, ...]
        ta_final_result, ta2_final_result = resize_image(img_temp, aspect_ratio, input_size, M, N, newH, newW, i)
        ta_final_result = resize_image_float(ta_final_result, tf.constant([1., 0, 0, 0, 1., 0]), out_dims=out_dims)
        ta2_final_result = resize_image_float(ta2_final_result, tf.constant([1., 0, 0, 0, 1., 0]), out_dims=out_dims2)
        return [tf.add(i, 1), img.write(i, ta_final_result), img2.write(i, ta2_final_result)]

    n, img_final, img_final2 = tf.while_loop(condition, body, init_state)
    return img_final.stack(), img_final2.stack()


def resize_image(images, aspect_ratio, input_size, M, N, newH, newW, i):
    ta_final_result, ta2_final_result = tf.cond(tf.abs(aspect_ratio[0] - input_size[1]) > 1,
                              lambda: resize_columns(images, M, newH, i),
                              lambda: (images, images))

    resized_images = tf.transpose(ta_final_result, [1, 0, 2])

    ta_final_result = tf.cond(tf.abs(aspect_ratio[1] - input_size[2]) > 1,
                              lambda: resize_columns_forward(resized_images, N, newW, i),
                              lambda: resized_images)

    ta_final_result = tf.transpose(ta_final_result, [1, 0, 2])

    # reverse image
    resized2_images = tf.transpose(ta2_final_result, [1, 0, 2])

    ta2_final_result = tf.cond(tf.abs(aspect_ratio[1] - input_size[2]) > 1,
                              lambda: resize_columns_reverse(resized2_images, N, newW, i),
                              lambda: resized2_images)

    ta2_final_result = tf.transpose(ta2_final_result, [1, 0, 2])

    return ta_final_result, ta2_final_result


def resize_columns(images, M, newH, i):
    ta = tf.TensorArray(dtype=tf.float32, size=M, infer_shape=False)
    ta2 = tf.TensorArray(dtype=tf.float32, size=M, infer_shape=False)
    new_size = tf.subtract(newH[i, 1:], newH[i, 0:-1])
    theta = tf.constant([1., 0, 0, 0, 1., 0])

    init_state = (0, ta, ta2)
    condition = lambda j, _, _j: j < M

    def body(j, ta, ta2):
        block = images[j * grid_size:(j + 1) * grid_size, :, :]
        target_shape = tf.cast(tf.shape(images), tf.float32)
        out_dims = [new_size[j], target_shape[1], target_shape[2]]

        new_block = tf.cond(new_size[j] > 0,
                lambda: resize_image_float(block, theta, out_dims=out_dims),
                lambda: tf.zeros((0, tf.shape(images)[1], tf.shape(images)[2])))

        new_block_reverse = tf.cond(new_size[j] > 0,
                lambda: resize_image_float(new_block, theta, out_dims=[grid_size*1.0, target_shape[1], target_shape[2]]),
                lambda: tf.zeros((0, tf.shape(images)[1], tf.shape(images)[2])))

        return [tf.add(j, 1), ta.write(j, new_block), ta2.write(j, new_block_reverse)]

    n, ta_final, ta2_final = tf.while_loop(condition, body, init_state)
    return ta_final.concat(), ta2_final.concat()


def resize_columns_forward(images, M, newH, i):
    ta = tf.TensorArray(dtype=tf.float32, size=M, infer_shape=False)
    new_size = tf.subtract(newH[i, 1:], newH[i, 0:-1])
    theta = tf.constant([1., 0, 0, 0, 1., 0])

    init_state = (0, ta)
    condition = lambda j, _: j < M

    def body(j, ta):
        block = images[j * grid_size:(j + 1) * grid_size, :, :]
        target_shape = tf.cast(tf.shape(images), tf.float32)
        out_dims = [new_size[j], target_shape[1], target_shape[2]]

        new_block = tf.cond(new_size[j] > 0,
                lambda: resize_image_float(block, theta, out_dims=out_dims),
                lambda: tf.zeros((0, tf.shape(images)[1], tf.shape(images)[2])))
        return [tf.add(j, 1), ta.write(j, new_block)]

    n, ta_final = tf.while_loop(condition, body, init_state)
    return ta_final.concat()


def resize_columns_reverse(images, M, newH, i):
    ta = tf.TensorArray(dtype=tf.float32, size=M, infer_shape=False)
    new_size = tf.subtract(newH[i, 1:], newH[i, 0:-1])
    theta = tf.constant([1., 0, 0, 0, 1., 0])

    init_state = (0, ta)
    condition = lambda j, _: j < M

    def body(j, ta):
        block = images[j * grid_size:(j + 1) * grid_size, :, :]
        target_shape = tf.cast(tf.shape(images), tf.float32)
        out_dims = [new_size[j], target_shape[1], target_shape[2]]

        new_block = tf.cond(new_size[j] > 0,
                lambda: resize_image_float(block, theta, out_dims=out_dims),
                lambda: tf.zeros((0, tf.shape(images)[1], tf.shape(images)[2])))

        new_block = tf.cond(new_size[j] > 0,
                lambda: resize_image_float(new_block, theta,
                                           out_dims=[grid_size*1.0, target_shape[1], target_shape[2]]),
                lambda: tf.zeros((0, tf.shape(images)[1], tf.shape(images)[2])))

        return [tf.add(j, 1), ta.write(j, new_block)]

    n, ta_final = tf.while_loop(condition, body, init_state)
    return ta_final.concat()


def reconstruct_image(images, featH, featW, input_size, aspect_ratio):
    newH, newW, B, M, N, input_size = preprocess(featH, featW, input_size, aspect_ratio)
    ta_final_result, ta2_final_result = resize_images(images, aspect_ratio, input_size, B, M, N, newH, newW)
    return newH, newW, ta_final_result, ta2_final_result


if __name__ == '__main__':
    img = imread('./RetargetMeAll/boat.png')
    H, W, _ = img.shape
    print(img.shape)
    img = imresize(img, (H-np.mod(H,grid_size), W-np.mod(W,grid_size)))
    img = np.stack((img, img, img, img, img, img))
    B, H, W, C = img.shape

    print(H/grid_size)
    print(W/grid_size)
    print(img.shape)

    images = tf.placeholder(tf.float32, shape=(B, H, W, C))
    featH = tf.Variable(np.random.normal(0, 1, (B, H/grid_size)), dtype=tf.float32)
    featH = tf.abs(featH)
    featW = tf.Variable(np.random.normal(0.5, 1, (B, W/grid_size)), dtype=tf.float32)
    featW = tf.abs(featW)
    input_size = tf.shape(images)
    aspect_ratio = tf.constant([np.round(H / 1.0), W / 2])

    featH = tf.nn.sigmoid(featH)
    # featH = tf.clip_by_value(featH, 0, 1.0)
    featW = tf.nn.sigmoid(featW)
    # featW = tf.clip_by_value(featW, 0, 1.0)

    newH, newW, ta_final_result, ta2_final_result = reconstruct_image(images, featH, featW, input_size, aspect_ratio)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Runs the op.
        x, x2, y, z = sess.run([ta_final_result, ta2_final_result, newH[0,:], newW[0,:]], feed_dict={images: img})

        x = np.array(x)
        imshow(x[0,...])
        print(x.shape)
        x2 = np.array(x2)
        imshow(x2[0, ...])
        print(x2.shape)
        print(y, z)
