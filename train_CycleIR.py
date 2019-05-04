# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from VGG_MODEL import vgg16 as vgg16
from parameter_define import *
from scipy.misc import imresize, imread, imshow, imsave
from TMM_circle_reconstruct_image import reconstruct_image
from data_loader import data_loader
import glob


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator_conv(images, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()

        with tf.name_scope("generator_vgg16"):
            vgg = vgg16.Vgg16('./VGG_MODEL/vgg16.npy')
            vgg.build(images)

        vgg_map = vgg.conv4_1

        # primary channel
        train = ly.conv2d(vgg_map, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        train = ly.conv2d(train, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        # train = ly.max_pool2d(inputs=train, kernel_size=2, stride=2, padding='SAME')
        train = ly.conv2d(train, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        # spatial attention
        SPAttention1 = ly.conv2d(train, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        SPAttention2 = ly.conv2d(SPAttention1, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        SPAttention3 = ly.conv2d(SPAttention2, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.sigmoid)

        SPAtrain = train * SPAttention3

        # channel attention
        CHAttention = ly.conv2d(SPAttention2, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        CHAttention = tf.reduce_mean(CHAttention, [1, 2])
        CHAttention = tf.layers.dense(CHAttention, 32, tf.nn.sigmoid)

        # final map
        salmap = tf.reduce_mean(SPAtrain * tf.reshape(CHAttention, [tf.shape(images)[0], 1, 1, 32]), 3)

        final_max = tf.tile(tf.reduce_max(salmap, [1, 2], keep_dims=True), [1, tf.shape(salmap)[1], tf.shape(salmap)[2]])
        salmap = salmap / final_max

        # height
        featH = tf.squeeze(tf.reduce_mean(salmap, 2))
        featH = tf.reshape(featH, [tf.shape(images)[0], -1])+0.1

        # width
        featW = tf.squeeze(tf.reduce_mean(salmap, 1))
        featW = tf.reshape(featW, [tf.shape(images)[0], -1])+0.1

    return featH, featW, salmap


def salmap_cal(images):
    with tf.name_scope("generator_vgg16"):
        vgg = vgg16.Vgg16('./VGG_MODEL/vgg16.npy')
        vgg.build(images)

    shape = tf.shape(images)
    # conv1_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv1_1, 3, True), [shape[1], shape[2]])
    # conv1_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv1_2, 3, True), [shape[1], shape[2]])
    # conv2_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv2_1, 3, True), [shape[1], shape[2]])
    # conv2_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv2_2, 3, True), [shape[1], shape[2]])
    # conv3_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv3_1, 3, True), [shape[1], shape[2]])
    # conv3_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv3_2, 3, True), [shape[1], shape[2]])
    # conv3_3 = tf.image.resize_images(tf.reduce_mean(vgg.conv3_3, 3, True), [shape[1], shape[2]])
    conv4_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv4_1, 3, True), [shape[1], shape[2]])
    conv4_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv4_2, 3, True), [shape[1], shape[2]])
    conv4_3 = tf.image.resize_images(tf.reduce_mean(vgg.conv4_3, 3, True), [shape[1], shape[2]])
    # conv4_1 = tf.reduce_mean(vgg.conv4_1, 3, True)
    # conv4_2 = tf.reduce_mean(vgg.conv4_2, 3, True)
    # conv4_3 = tf.reduce_mean(vgg.conv4_3, 3, True)
    # shape = tf.shape(vgg.conv4_1)
    conv5_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv5_1, 3, True), [shape[1], shape[2]])
    conv5_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv5_2, 3, True), [shape[1], shape[2]])
    conv5_3 = tf.image.resize_images(tf.reduce_mean(vgg.conv5_3, 3, True), [shape[1], shape[2]])
    final_sal_map = conv4_1 + conv4_2 + conv4_3 + (conv5_1 + conv5_2 + conv5_3) * 3
    return final_sal_map


def salmap_norm(final_sal_map):
    # normalize final_sal_map.
    temp_shape = tf.shape(final_sal_map)
    # final_min = tf.tile(tf.reduce_min(final_sal_map, [1, 2], keep_dims=True), [1, temp_shape[1], temp_shape[2]])
    final_max = tf.tile(tf.reduce_max(final_sal_map, [1, 2, 3], keep_dims=True), [1, temp_shape[1], temp_shape[2], temp_shape[3]])
    salmap = final_sal_map / final_max
    # final_sal_map = (final_sal_map - final_min) / (final_max - final_min)
    return salmap


def build_graph():
    aspect_ratio = tf.placeholder(dtype=tf.float32, shape=[2,])
    input_size = tf.placeholder(dtype=tf.int32, shape=[4,])  #BHWC
    images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    sals = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])

    input_ = tf.cast(input_size, tf.float32)
    aspect_ratio_HR = [input_[1] * input_[1] / aspect_ratio[0], input_[2] * input_[2] / aspect_ratio[1]]
    aspect_ratio_HR = tf.floor(aspect_ratio_HR)

    # first output
    featH, featW, salmap = generator_conv(images, reuse=False)
    images_LR = reconstruct_image(images, featH, featW, input_size, aspect_ratio)
    images_HR = reconstruct_image(images, (2-featH), (2-featW), input_size, aspect_ratio_HR)

    # LR to HR
    aspect_LR = tf.cast(aspect_ratio, tf.int32)
    aspect_LR = [input_size[0], aspect_LR[0], aspect_LR[1], 3]
    images_LR = tf.reshape(images_LR, aspect_LR)
    featH_LR, featW_LR, salmap_LR = generator_conv(images_LR, reuse=True)
    images_HR_from_LR = reconstruct_image(images_LR, (2-featH_LR), (2-featW_LR), aspect_LR, [input_[1], input_[2]])

    # HR to LR
    aspect_HR = tf.cast(aspect_ratio_HR, tf.int32)
    aspect_HR = [input_size[0], aspect_HR[0], aspect_HR[1], 3]
    images_HR = tf.reshape(images_HR, aspect_HR)
    featH_HR, featW_HR, salmap_HR = generator_conv(images_HR, reuse=True)
    images_LR_from_HR = reconstruct_image(images_HR, featH_HR, featW_HR, aspect_HR, [input_[1], input_[2]])

    Input_sal_map = salmap_cal(images)
    InputLR_sal_map = salmap_cal(images_HR_from_LR)
    InputHR_sal_map = salmap_cal(images_LR_from_HR)

    # sals = sals / 255.0
    # shape = tf.shape(images)
    # weight_sal = tf.image.resize_images(Input_sal_map, [shape[1], shape[2]])

    g_loss = tf.reduce_mean(tf.squared_difference(InputLR_sal_map, Input_sal_map))+ \
             tf.reduce_mean(tf.squared_difference(InputHR_sal_map, Input_sal_map))

    # g_loss = tf.reduce_mean(tf.squared_difference(InputHR_sal_map, Input_sal_map)) + \
    #          tf.reduce_mean(tf.squared_difference(InputLR_sal_map, Input_sal_map)) + \
    #          tf.reduce_mean(tf.squared_difference(images, images_HR_from_LR)) + \
    #          tf.reduce_mean(tf.squared_difference(images, images_LR_from_HR))

    images_LR_from_HR_sum = tf.summary.image("images_HR_from_LR", images_HR_from_LR)
    images_HR_sum = tf.summary.image("images_HR", images_HR)
    images_LR_sum = tf.summary.image("images_LR", images_LR)
    images_LR_from_HR_sum = tf.summary.image("images_LR_from_HR", images_LR_from_HR)
    images_input_sum = tf.summary.image("images", images)
    sals_GT_sum = tf.summary.image("sals_GT", sals * 255)
    salmap_sum = tf.summary.image("salmap", tf.expand_dims(salmap, 3) * 255)
    Input_sal_map_sum = tf.summary.image("Input_sal_map", Input_sal_map * 255)
    InputLR_sal_map_sum = tf.summary.image("InputLR_sal_map", InputLR_sal_map * 255)
    InputHR_sal_map_sum = tf.summary.image("InputHR_sal_map", InputHR_sal_map * 255)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
                             optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer,
                             variables=theta_g, global_step=counter_g)

    return sals, opt_g, aspect_ratio, images, g_loss, images_LR, salmap, input_size, featH, featW


def model_test(sess, images, salmap, images_new, train_step, aspect_ratio, input_size):
    img_fnames = [os.path.join(dirpath, x) for x in os.listdir(dirpath) if x.endswith(img_ext)]
    base_fnames = [os.path.splitext(os.path.basename(x))[0] + img_ext for x in img_fnames]

    for i, (imgPath, imgName) in enumerate(zip(img_fnames, base_fnames)):
        img = imread(imgPath)
        img = np.expand_dims(img, 0)
        shape = img.shape
        feed_dict = {images: img, aspect_ratio: [shape[1], shape[2]/2], input_size: shape}
        [images_new_, salmap_] = sess.run([images_new, salmap], feed_dict=feed_dict)
        images_new_ = np.array(images_new_, np.uint8)
        salmap_ = np.array(salmap_, np.float32)
        salmap_ = (salmap_ - np.min(salmap_))/(np.max(salmap_) - np.min(salmap_))*255
        salmap_ = np.array(salmap_, np.uint8)
        imsave(outpath + '/step' + str(train_step) + '_img_' + imgName, images_new_[0, ...])
        imsave(outpath + '/step' + str(train_step) + '_sal_' + imgName, imresize(salmap_[0,:,:], [shape[1], shape[2]], 'bicubic'))


def main(argv=None):
    checkpoint_path = '/data/TF_project/circle_IR/ckpt_wgan'
    CHECKPOINT_FILE = checkpoint_path + "/checkpoint.ckpt"
    LATEST_CHECKPOINT = tf.train.latest_checkpoint(checkpoint_path)

    def restore_from_checkpoint(sess, saver, checkpoint):
        if checkpoint:
            print("Restore session from checkpoint: {}".format(checkpoint))
            saver.restore(sess, checkpoint)
            return True
        else:
            print("Checkpoint not found: {}".format(checkpoint))
            return False

    sals, opt_g, aspect_ratio, images, g_loss, images_LR, salmap, input_size, featH, featW = build_graph()

    saver = tf.train.Saver()
    merged_all = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('initial successfully!')

        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)

        max_iter_step = int(num_epochs * len(glob.glob('training_dataset/HKU-IS/gt/*.png'))) // batch_size
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        dl = data_loader()
        imgs, gts = dl.get_data(batch_size=batch_size, shuffle_num=1000, prefetch_num=1000)

        for i in range(max_iter_step):
            # ---------- Generator training ----------
            img, gt = sess.run([imgs, gts])

            new_height = np.float32(np.random.randint(img_height / 4, img_height / 2))
            new_width = np.float32(np.random.randint(img_width / 4, img_width / 2))
            aspect_ratio_ = [new_height, new_width]

            input_size_ = [batch_size, img_height, img_width, 3]
            feed_dict = {images: img, sals: gt, aspect_ratio: aspect_ratio_, input_size: input_size_}

            if i % 25 == 24:
                _, merged, generator_loss, featH_, featW_ = sess.run([opt_g, merged_all, g_loss, featH, featW], feed_dict=feed_dict)
                summary_writer.add_summary(merged, i)
                print(featH_[0, :])
                print('featH mean: ', np.mean(featH_, 1))
                print(featW_[0, :])
                print('featW mean: ', np.mean(featW_, 1))
            else:
                _, generator_loss = sess.run([opt_g, g_loss], feed_dict=feed_dict)

            print("Generator training step {}/{}, generator_loss: {}".format(i, max_iter_step, generator_loss))

            if i % 400 == 399:
                model_test(sess, images, salmap, images_LR, i, aspect_ratio, input_size)

            if i % 900 == 899:
                saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=max_iter_step)


if __name__ == "__main__":
    tf.app.run()
