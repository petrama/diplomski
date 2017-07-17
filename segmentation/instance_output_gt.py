import tensorflow as tf
import read_cityscapes_tf_records as reader
import train_helper
import time
import os

import PIL
from PIL import Image

import skimage.transform
import skimage

import eval_helper
import numpy as np

import cv2

import helper

import sys

from shutil import copyfile


output_dir='/home/pmarce/datasets/cityscapes/gt_resized/val'

clustering_output='/home/pmarce/datasets/cityscapes/clustering/val/'

import matplotlib.pyplot as plt
tf.app.flags.DEFINE_string('config_path', "config/instance_output.py", """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.__dict__['__flags'].keys())

hasInstances={-1: False,
 0: False,
 1: False,
 2: False,
 3: False,
 4: False,
 5: False,
 6: False,
 7: False,
 8: False,
 9: False,
 10: False,
 11: True,
 12: True,
 13: True,
 14: True,
 15: True,
 16: True,
 17: True,
 18: True,
 255: True}

train_id_to_city_id={0:7,
                     1:8,
                     2:11,
                     3:12,
                     4:13,
                     5:17,
                     6:19,
                     7:20,
                     8:21,
                     9:22,
                     10:23,
                     11:24,
                     12:25,
                     13:26,
                     14:27,
                     15:28,
                     16:31,
                     17:32,
                     18:33,
                     255:-1}





class_names=['road',#1
             'sidewalk',#2,
             'building',#3
             'wall',#4
             'fence',#5
             'pole',#6
             'traffic light',#7,
             'traffic sign',#8
             'vegetation',#9
             'terrain',#10
             'sky',#11
             'person',#12
             'rider',#13
             'car',#14
             'truck',#15
             'bus',#16
             'train',#17
             'motorcycle',#18
             'bicycle']#19


from pyclustering.cluster.optics import optics
from pyclustering.utils import  timedcall;



img_shape=(288,640)
def optics_clustering(data, eps, minpts, amount_clusters=None):
    optics_instance = optics(data, eps, minpts, amount_clusters, ccore=True);
    (ticks, _) = timedcall(optics_instance.process);
    clusters = optics_instance.get_clusters();
    print(ticks)
    print('Number of clusters: ', len(clusters))
    print(sum(len(c) for c in clusters))
    return clusters, optics_instance

def visualize_clusters(clusters,mask_for_clustering):
    maska_clusters=np.zeros(np.sum(mask_for_clustering))
    for i in range(len(clusters)):
        maska_clusters[clusters[i]]=i+1
    final_mask=np.zeros(img_shape)*-1
    final_mask[np.where(mask_for_clustering)]=maska_clusters

    #print(final_mask[final_mask!=0])
    #plt.imshow(final_mask);plt.show()
    return final_mask

def softmax(x):

    expx=np.exp(x)
    sumexp=np.sum(expx)
    #sumexp[sumexp==0]=1e-8
    return expx/sumexp

def get_filtering_mask_by_segmentation_output(vlog,confidence_threshold=0.55):
    predicted_labels = vlog.argmax(2).astype(np.int32, copy=False).reshape(img_shape)
    confidences=np.apply_along_axis(softmax,2,vlog).max(2)
    #plt.figure();plt.imshow(predicted_labels)
    #plt.figure();plt.imshow(confidences)
    lab_has_inst=np.vectorize(hasInstances.get)(predicted_labels)
    lab_hasnt_inst=~lab_has_inst
    confident=confidences>confidence_threshold
    mask_not_clustering=lab_hasnt_inst * confident
    mask_clustering=~mask_not_clustering
    return mask_clustering,confidences


def train(model, resume_path=None):
    with tf.Graph().as_default():
        # configure the training session
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=config)

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        num_batches_per_epoch = (FLAGS.train_size //
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)




        val_data, val_labels, val_instances, val_instance_mask,val_vector_centroid,val_img_name = reader.inputs(shuffle=False,
                                                                     num_epochs=1,
                                                                     dataset_partition='val')



        with tf.variable_scope('model',reuse=None):
            logits_val,xss_val_node, loss_val,loss_ce_val,loss_reg_val = model.build(val_data, val_labels,None,
                                                                      val_vector_centroid, val_instance_mask,is_training=False)




        saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_epochs,sharded=False)



        if len(FLAGS.resume_path) > 0:
            print('\nRestoring params from:', FLAGS.resume_path)

            saver.restore(sess, FLAGS.resume_path)
            sess.run(tf.local_variables_initializer())


        else:
            print('No resume path!')
            return



        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        y, x = np.mgrid[0:FLAGS.img_height, 0:FLAGS.img_width]
        grid = np.stack((y, x), axis=-1).astype(float)

        num_batches = FLAGS.val_size // FLAGS.batch_size

        rmse=0



        for step in range(1, num_batches + 1):
            print(step)
            run_ops = [val_img_name,val_labels,val_instances,val_instance_mask,val_vector_centroid,xss_val_node,logits_val]
            ret_val = sess.run(run_ops)
            name,labels,instances,ins_mask,gt_vec,xss_val,vlog=ret_val

            name=name[0].decode("utf-8")
            town=name.split('_')[0]



            labs=np.squeeze(labels)
            print(np.unique(labs))
            print(np.unique(train_id_to_city_id.keys()))
            labs=np.vectorize(train_id_to_city_id.get)(labs).astype(np.uint8)
            labs=labs.reshape(FLAGS.img_height,FLAGS.img_width)

            ins=np.squeeze(instances)

            ins=ins.reshape(FLAGS.img_height,FLAGS.img_width)

            im = Image.fromarray(ins.astype(np.uint16).astype(np.int32))
            print(np.unique(labs))
            print(np.unique(ins))
            out_dir=os.path.join(output_dir,town)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)



            im.save(os.path.join(out_dir,'{}_gtFine_instanceIds.png'.format(name)))

            #break


        coord.request_stop()
        coord.join(threads)
        sess.close()







def main(argv=None):
    model = helper.import_module('model', FLAGS.model_path)
    train(model)


if __name__ == '__main__':
    tf.app.run()
