import tensorflow as tf


def matrix_iou(a,b):
    ae = tf.expand_dims(a, 1)
    lt = tf.maximum(ae[:, :, :2], b[:, :2])
    rb = tf.minimum(ae[:, :, 2:], b[:, 2:])
    difference = rb - lt
    difference = tf.where(tf.greater(difference, 0), difference, tf.zeros_like(difference))
    area_i = tf.reduce_prod(difference, axis=2)
    area_a = tf.reduce_prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = tf.reduce_prod(b[:, 2:] - b[:, :2], axis=1)
    iou = area_i / (tf.expand_dims(area_a, 1) + area_b - area_i)
    return iou
