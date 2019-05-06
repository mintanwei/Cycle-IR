# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from VGG_MODEL import vgg16 as vgg16
from parameter_define import *
from scipy.misc import imresize, imread, imsave
from Circle_reconstruct_image import reconstruct_image
import time


def CycleIR(images, reuse=False):
    with tf.variable_scope('CycleIR') as scope:
        if reuse:
            scope.reuse_variables()

        # backbone
        with tf.name_scope("CycleIR_vgg16"):
            vgg = vgg16.Vgg16('./VGG_MODEL/vgg16.npy')
            vgg.build(images)

        vgg_map = vgg.conv4_1
        train = ly.conv2d(vgg_map, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        train = ly.conv2d(train, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
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

        # attention map
        salmap = tf.reduce_mean(SPAtrain * tf.reshape(CHAttention, [tf.shape(images)[0], 1, 1, 32]), 3)
        final_max = tf.tile(tf.reduce_max(salmap, [1, 2], keep_dims=True), [1, tf.shape(salmap)[1], tf.shape(salmap)[2]])
        AttentionMap = salmap / final_max

        # Sh and Sw
        Sh = tf.squeeze(tf.reduce_mean(AttentionMap, 2))
        Sh = tf.reshape(Sh, [tf.shape(images)[0], -1])+0.1
        Sw = tf.squeeze(tf.reduce_mean(AttentionMap, 1))
        Sw = tf.reshape(Sw, [tf.shape(images)[0], -1])+0.1
    return Sh, Sw, AttentionMap


def PerceptualCompute(images):
    with tf.name_scope("PerceptualCompute_vgg16"):
        vgg = vgg16.Vgg16('./VGG_MODEL/vgg16.npy')
        vgg.build(images)

    shape = tf.shape(images)
    conv4_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv4_1, 3, True), [shape[1], shape[2]])
    conv4_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv4_2, 3, True), [shape[1], shape[2]])
    conv4_3 = tf.image.resize_images(tf.reduce_mean(vgg.conv4_3, 3, True), [shape[1], shape[2]])
    conv5_1 = tf.image.resize_images(tf.reduce_mean(vgg.conv5_1, 3, True), [shape[1], shape[2]])
    conv5_2 = tf.image.resize_images(tf.reduce_mean(vgg.conv5_2, 3, True), [shape[1], shape[2]])
    conv5_3 = tf.image.resize_images(tf.reduce_mean(vgg.conv5_3, 3, True), [shape[1], shape[2]])
    PerceptualMap = conv4_1 + conv4_2 + conv4_3 + (conv5_1 + conv5_2 + conv5_3) * 3
    return PerceptualMap


def build_graph():
    aspect_ratio = tf.placeholder(dtype=tf.float32, shape=[2,])
    input_size = tf.placeholder(dtype=tf.int32, shape=[4,])  #BHWC
    images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

    input_ = tf.cast(input_size, tf.float32)
    aspect_ratio_HR = [input_[1] * input_[1] / aspect_ratio[0], input_[2] * input_[2] / aspect_ratio[1]]
    aspect_ratio_HR = tf.floor(aspect_ratio_HR)

    # Forward retargeting
    Sh, Sw, AttentionMap = CycleIR(images, reuse=False)
    images_LR = reconstruct_image(images, Sh, Sw, input_size, aspect_ratio)
    images_HR = reconstruct_image(images, (2-Sh), (2-Sw), input_size, aspect_ratio_HR)

    # Reverse retargeting (top)
    aspect_LR = tf.cast(aspect_ratio, tf.int32)
    aspect_LR = [input_size[0], aspect_LR[0], aspect_LR[1], 3]
    images_LR = tf.reshape(images_LR, aspect_LR)
    featH_LR, featW_LR, salmap_LR = CycleIR(images_LR, reuse=True)
    images_HR_from_LR = reconstruct_image(images_LR, (2-featH_LR), (2-featW_LR), aspect_LR, [input_[1], input_[2]])

    # Reverse retargeting (bottom)
    aspect_HR = tf.cast(aspect_ratio_HR, tf.int32)
    aspect_HR = [input_size[0], aspect_HR[0], aspect_HR[1], 3]
    images_HR = tf.reshape(images_HR, aspect_HR)
    featH_HR, featW_HR, salmap_HR = CycleIR(images_HR, reuse=True)
    images_LR_from_HR = reconstruct_image(images_HR, featH_HR, featW_HR, aspect_HR, [input_[1], input_[2]])

    PerceptualMap_Input = PerceptualCompute(images)
    PerceptualMap_RevTop = PerceptualCompute(images_HR_from_LR)
    PerceptualMap_RevBom = PerceptualCompute(images_LR_from_HR)

    CycleLoss = tf.reduce_mean(tf.squared_difference(PerceptualMap_Input, PerceptualMap_RevTop)) + \
                tf.reduce_mean(tf.squared_difference(PerceptualMap_Input, PerceptualMap_RevBom))

    # CycleLoss = tf.reduce_mean(tf.squared_difference(PerceptualMap_Input, PerceptualMap_RevTop)) + \
    #             tf.reduce_mean(tf.squared_difference(PerceptualMap_Input, PerceptualMap_RevBom)) + \
    #             tf.reduce_mean(tf.squared_difference(images, images_HR_from_LR)) + \
    #             tf.reduce_mean(tf.squared_difference(images, images_LR_from_HR))

    tf.summary.image("images_HR", images_HR)
    tf.summary.image("images_LR", images_LR)
    tf.summary.image("images", images)
    tf.summary.image("AttentionMap", tf.expand_dims(AttentionMap, 3) * 255)
    tf.summary.image("PerceptualMap_Input", PerceptualMap_Input * 255)
    tf.summary.image("PerceptualMap_RevTop", PerceptualMap_RevTop * 255)
    tf.summary.image("PerceptualMap_RevBom", PerceptualMap_RevBom * 255)

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CycleIR')
    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    opt_g = ly.optimize_loss(loss=CycleLoss, learning_rate=learning_rate_ger,
                             optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer,
                             variables=theta_g, global_step=counter_g)

    return opt_g, aspect_ratio, images, CycleLoss, images_LR, AttentionMap, input_size, Sh, Sw


def model_test(sess, images, salmap, images_new, train_step, aspect_ratio, input_size):
    dirpath = 'test_image/'  # test input path
    img_ext = '.jpg'
    outpath = './test_image_result/'  # test output path

    img_fnames = [os.path.join(dirpath, x) for x in os.listdir(dirpath) if x.endswith(img_ext)]
    base_fnames = [os.path.splitext(os.path.basename(x))[0] + img_ext for x in img_fnames]

    for i, (imgPath, imgName) in enumerate(zip(img_fnames, base_fnames)):
        img = imread(imgPath)
        img = np.expand_dims(img, 0)
        shape = img.shape
        target_ratio = ((shape[1], shape[2] * 0.5), (shape[1], shape[2] * 0.75))

        for TR in target_ratio:
            feed_dict = {images: img, aspect_ratio: TR, input_size: shape}
            [images_new_, salmap_] = sess.run([images_new, salmap], feed_dict=feed_dict)
            images_new_ = np.array(images_new_, np.uint8)
            salmap_ = np.array(salmap_, np.float32)
            salmap_ = (salmap_ - np.min(salmap_))/(np.max(salmap_) - np.min(salmap_))*255
            salmap_ = np.array(salmap_, np.uint8)
            imsave(outpath + str(TR) + '_img_' + imgName, images_new_[0, ...])
            imsave(outpath + str(TR) + '_sal_' + imgName, imresize(salmap_[0,:,:], [shape[1], shape[2]], 'bicubic'))


def restore_from_checkpoint(sess, saver, checkpoint):
    if checkpoint:
        print("Restore session from checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
        return True
    else:
        print("Checkpoint not found: {}".format(checkpoint))
        return False


def main(argv=None):
    checkpoint_path = './ckpt_wgan'
    CHECKPOINT_FILE = checkpoint_path + "/checkpoint.ckpt"
    LATEST_CHECKPOINT = tf.train.latest_checkpoint(checkpoint_path)

    opt_g, aspect_ratio, images, CycleLoss, images_LR, AttentionMap, input_size, Sh, Sw = build_graph()

    saver = tf.train.Saver()
    merged_all = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('initial successfully!')

        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
        model_test(sess, images, AttentionMap, images_LR, 0, aspect_ratio, input_size)


if __name__ == "__main__":
    tf.app.run()
