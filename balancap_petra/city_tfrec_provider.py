# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/data/create_cifar10_dataset.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import glob

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

_FILE_PATTERN = '*.tfrecords'

SPLITS_TO_SIZES = {'train': 2975, 'val': 500}

_NUM_CLASSES = 9


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([1], tf.int64),
            'width': tf.FixedLenFeature([1], tf.int64),
            'depth': tf.FixedLenFeature([1], tf.int64),

            'img_name': tf.FixedLenFeature((), tf.string, default_value=''),

            'rgb': tf.FixedLenFeature((), tf.string, default_value=''),
            'labels': tf.FixedLenFeature((), tf.string, default_value=''),
            'instances': tf.FixedLenFeature((), tf.string, default_value=''),
            'vector_centroid': tf.FixedLenFeature((), tf.string, default_value=''),
            'instance_mask': tf.FixedLenFeature((), tf.string, default_value=''),

            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),

        })

    # img_shape=tf.stack([img_height,img_width,num_channels],axis=-1)

    img_name=features['img_name']
    image = tf.decode_raw(features['rgb'], tf.uint8)
    #labels_unary = tf.decode_raw(features['labels'], tf.uint8)
    #instances_unary = tf.decode_raw(features['instances'], tf.int16)
    #instance_masku = tf.decode_raw(features['instance_mask'], tf.uint8)

    vec_cen = tf.decode_raw(features['vector_centroid'], tf.float32)

    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])

    bbox_labels = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
    bbox_labels = tf.where(tf.equal(bbox_labels, 255),
                           tf.ones_like(bbox_labels) * 19, bbox_labels)
    bbox_labels = bbox_labels - tf.ones_like(bbox_labels) * 10

    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    image = tf.reshape(image, shape=(FLAGS.img_height,FLAGS.img_width,FLAGS.num_channels))
    #image = tf.reshape(image, shape=(288,640,3))
    #image = tf.to_float(image)

    #num_pixels = FLAGS.img_height * FLAGS.img_width
    #labels = tf.reshape(labels_unary, shape=tf.reshape(num_pixels, [-1]))
    ##labels = tf.to_float(labels)
    #labels = tf.cast(labels, tf.int32)

    #instances = tf.reshape(instances_unary, shape=tf.reshape(num_pixels, [-1]))
    #instance_mask = tf.reshape(instance_masku, shape=tf.reshape(num_pixels, [-1]))

    #vector_centroid = tf.reshape(vec_cen, shape=tf.stack([FLAGS.img_height, FLAGS.img_width, 2], axis=-1))

    #return (image, labels, instances, instance_mask, vector_centroid,
    #        bboxes, bbox_labels)

    return image,bboxes,bbox_labels,img_name

def get_filenames(split_name, dataset_dir, file_pattern=None):
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, split_name, file_pattern)
    to=sorted(glob.glob(file_pattern))
    print(file_pattern)
    print(to)
    return to


def inputs(dataset_dir,shuffle=True, num_epochs=False, dataset_partition='train'):
    if not num_epochs:
        num_epochs = None

    files = get_filenames(dataset_partition,dataset_dir)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs,
                                                        shuffle=shuffle,
                                                        capacity=len(files))

        return read_and_decode(filename_queue)







