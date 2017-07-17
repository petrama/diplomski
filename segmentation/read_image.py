import skimage as ski
import skimage.transform
import skimage.data

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def read_and_resize(rgb_path):
    img_width, img_height = (FLAGS.img_width, FLAGS.img_height)
    rgb = ski.data.load(rgb_path)
    rgb = ski.transform.resize(
        rgb, (img_height, img_width), order=3)
    rgb = skimage.img_as_ubyte(rgb)
    rgb.resize((1, img_height, img_width, 3))
    return rgb

