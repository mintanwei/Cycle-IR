import tensorflow as tf
import numpy as np
from PIL import Image
from get_filename import FileNameDirectory
from parameter_define import *


############################################################
# read image file using queue manner
############################################################
class InputImages(FileNameDirectory):
    def __init__(self, file_dir1, image_height, image_width, image_channel, batch_size, model='file'):
        FileNameDirectory.__init__(self, file_dir1, model)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel

    # get imagename lists.
    def get_filename(self):
        self.image_names, self.label = self.fileName_model()

    def create_filename_queue(self):
        self.filename_queue = tf.train.string_input_producer(self.image_names, shuffle=False)

    def read_and_decode(self):
        reader = tf.WholeFileReader()
        key, value = reader.read(self.filename_queue)
        my_img = tf.image.decode_png(value, channels=self.image_channel, name='decode_image')
        # my_img = tf.image.random_flip_left_right(my_img)
        self.image = tf.cast(tf.image.resize_images(my_img, [self.image_height, self.image_width]), tf.uint8)
        print(self.image)

    def _generate_image_and_label_batch(self, min_queue_examples):
        """Construct a queued batch of images and labels."""
        num_preprocess_threads = 4
        image_batch = tf.train.shuffle_batch(
            [self.image],
            batch_size=self.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 5 * self.batch_size,
            min_after_dequeue=min_queue_examples)

        # # Display the training images in the visualizer.
        # tf.summary.image('read_images', image_batch)
        return image_batch

    def input_image(self):
        self.get_filename()
        self.create_filename_queue()
        self.read_and_decode()
        return self._generate_image_and_label_batch(min_queue_examples=int(0.01*len(self.image_names)))


def main():
    input = InputImages(img_dataset, 480, 640, 3, 16, model='file')
    image_batch = input.input_image()
    num_train_examples = len(input.image_names)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print('initial successfully!')

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_train_examples):  # length of your filename list
            image = sess.run(image_batch)  # here is your image Tensor :)
            print(image.shape)

            if i % 100 == 0:
                print(image.shape)
                image_import = Image.fromarray(np.asarray(image[0, ...]))
                image_import.show()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    print('begin...')
    main()
