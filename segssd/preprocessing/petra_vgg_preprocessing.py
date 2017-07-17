from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops

from preprocessing import tf_image
from nets import ssd_common

slim = tf.contrib.slim


# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

BBOX_CROP_OVERLAP=0.45

        # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25#0.25
CROP_RATIO_RANGE = (0.7, 1.3)

#sample_methods=IntEnum('sample_methods','ENTIRE_IMAGE',#Use the entire original input image.
#                                        'MIN_JACCARD',#Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,0.5, 0.7, or 0.9
#                                        'RANDOM')


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image

def use_entire_image(image,out_shape):
    bilinear_image = tf.image.resize_images(tf.expand_dims(image, 0), out_shape)
    bilinear_image=tf.cast(tf.floor(bilinear_image), tf.int32)
    return bilinear_image

def petra_preprocessing(image, labels, bboxes,out_shape, data_format):

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].

    method_selector=np.random.uniform()

    if method_selector>0.30: #crop!

        if method_selector > 0.6:
            image_pre, labels, bboxes, distort_bbox = \
                distorted_bounding_box_crop(image, labels, bboxes,
                                            min_object_covered=MIN_OBJECT_COVERED,
                                            aspect_ratio_range=CROP_RATIO_RANGE)

        else:
            min_object_covered=[0.1,0.5]
            ri=np.random.randint(0,len(min_object_covered))
            random_coverage=min_object_covered[ri]

            image_pre,labels,bboxes,distort_bbox=\
                distorted_big_bounding_box_crop(image,labels,bboxes,
                                                min_object_covered=random_coverage,
                                                aspect_ratio_range=CROP_RATIO_RANGE)

    else:
        image_pre=image #use entire image

    image_pre = tf.to_float(image_pre)
    image_mean_substracted = tf_image_whitened(image_pre, [_R_MEAN, _G_MEAN, _B_MEAN])

    image_pre = tf_image.resize_image(image_mean_substracted, out_shape,
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)

    image_pre, bboxes = tf_image.random_flip_left_right(image_pre, bboxes,seed=74)

    #mean = tf.constant([_R_MEAN,_G_MEAN,_B_MEAN], dtype=tf.int32)
    #image_mean_substracted=tf.cast(image_pre,tf.int32)-mean

    #image_mean_substracted=tf.cast(image_mean_substracted,tf.float32)


    return image_pre,labels,bboxes


def eval_preprocessing(image, labels, bboxes,out_shape, data_format):

    image = tf.to_float(image)
    image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    image_pre=tf_image.resize_image(image, out_shape,
                          method=tf.image.ResizeMethod.BILINEAR,
                          align_corners=False)

    #mean = tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], dtype=tf.int32)
    #image_mean_substracted = tf.cast(image_pre, tf.int32) - mean

    #image_mean_substracted = tf.cast(image_mean_substracted, tf.float32)
    image_mean_substracted = tf_image_whitened(image_pre, [_R_MEAN, _G_MEAN, _B_MEAN])

    return image_mean_substracted, labels, bboxes



def distorted_big_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered,
                                aspect_ratio_range,
                                area_range=(0.1, 1.0),
                                max_attempts=50):

    #big_box is box that encaptures all other bboxes
    big_box=tf.stack([tf.reduce_min(bboxes[:,0]),tf.reduce_min(bboxes[:,1]),tf.reduce_max(bboxes[:,2]),tf.reduce_max(bboxes[:,3])])
    big_box=tf.expand_dims(big_box,0)


    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(big_box,0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
    distort_bbox = distort_bbox[0, 0]

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    # Restore the shape since the dynamic slice loses 3rd dimension.
    cropped_image.set_shape([None, None, 3])

    # Update bounding boxes: resize and filter out.
    bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
    labels, bboxes = tfe.bboxes_filter_center(labels, bboxes)
    bboxes = tfe.bboxes_clip(tf.expand_dims(tf.constant([0., 0., 1., 1.]), 0), bboxes)
    return cropped_image, labels, bboxes, distort_bbox


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.9,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):

    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):


        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                #bounding_boxes=tf.expand_dims(big_box,0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_center(labels, bboxes)

        bboxes=tfe.bboxes_clip(tf.expand_dims(tf.constant([0.,0.,1.,1.]),0),bboxes)
        return cropped_image, labels, bboxes, distort_bbox

def preprocess_image(image,
                     labels,
                     bboxes,
                     out_shape,
                     data_format,
                     is_training=False,
                     **kwargs):

    if is_training:
        return petra_preprocessing(image, labels, bboxes, out_shape, data_format)
    else:
        return eval_preprocessing(image, labels, bboxes, out_shape, data_format)


