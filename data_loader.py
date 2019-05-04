import tensorflow as tf
import glob
import skimage.io as io
from scipy.misc import imresize
import numpy as np
import parameter_define


class data_loader():
    def generater(self):
        gts = sorted(glob.glob('training_dataset/HKU-IS/gt/*.png'))
        imgs = sorted(glob.glob('training_dataset/HKU-IS/imgs/*.png'))
        gts_len = len(gts)
        r = gts_len % self.batch_size
        if r != 0:
            r = self.batch_size - r
            gts = gts + gts[:r]
            imgs = imgs + gts[:r]

        height = parameter_define.img_height
        width = parameter_define.img_width
        for i in range(len(gts)):
            gt = io.imread(gts[i])
            img = io.imread(imgs[i])

            if(np.size(img.shape))<3:
                img = np.expand_dims(img, 2)
                img = np.concatenate([img, img, img], 2)

            gt = imresize(gt, [height, width], 'bicubic')
            img = imresize(img, [height, width], 'bicubic')

            gt = gt.astype(np.float32)
            gt = np.expand_dims(gt, -1)
            img = img.astype(np.float32)
            yield img, gt

    def get_data(self, batch_size=parameter_define.batch_size, shuffle_num=1000, prefetch_num=1000):
        self.batch_size = batch_size
        height = parameter_define.img_height
        width = parameter_define.img_width
        dataset = tf.data.Dataset.from_generator(self.generater, (tf.float32, tf.float32),
                                                 (tf.TensorShape([height, width, 3]), tf.TensorShape([height, width, 1])))
        dataset = dataset.shuffle(shuffle_num).prefetch(prefetch_num).batch(batch_size).repeat()
        return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    dl = data_loader()
    imgs, gts = dl.get_data()
    sess = tf.Session()

    for i in range(5000):
        o1, o2 = sess.run([imgs, gts])
        print(o1.shape, o2.shape)
        # io.imshow(o1[0] / 255.0)
        # io.show()
        # io.imshow(o2[0, :, :, 0] / 255.0)
        # io.show()
