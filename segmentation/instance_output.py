import tensorflow as tf
import read_cityscapes_tf_records as reader
import train_helper
import time
import os


from sklearn.cluster import DBSCAN

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



clustering_output='/home/pmarce/datasets/cityscapes/clustering_2/val/'

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

        start_time=time.time()

        for step in range(1, num_batches + 1):
            print(step)
            run_ops = [val_img_name,val_labels,val_instances,val_instance_mask,val_vector_centroid,xss_val_node,logits_val]
            ret_val = sess.run(run_ops)


            name,labels,instances,ins_mask,gt_vec,xss_val,vlog=ret_val

            name=name[0].decode("utf-8")
            town=name.split('_')[0]

            out_dir = os.path.join(clustering_output, town)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            ret=clustering(ins_mask,vlog,xss_val,grid)
            if ret:
                points,clusters,clu_labels,confidences,mask=ret

                output_file= os.path.join(clustering_output,town,name+'_outfile.txt')
                with open(output_file,'w') as f:


                    for i,cluster in enumerate(clusters):
                        #plt.figure()
                        mask_res=visualize_clusters([cluster],mask)

                        mask_file=os.path.join(out_dir, name + '_{}.png'.format(i))
                        print(' '.join((name + '_{}.png'.format(i),str(clu_labels[i]), str(confidences[i]))),file=f)

                        im = Image.fromarray(mask_res.astype(np.uint16).astype(np.int32))
                        im.save(mask_file)

            else:
                output_file = os.path.join(clustering_output, town, name + '_outfile.txt')
                with open(output_file, 'w') as f:
                    pass
                f.close()



            #break

        expired_time=time.time()-start_time

        print('Expired time: ',expired_time)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def clustering(ins_mask, vlog, xss_val, grid):

    xss_val = np.squeeze(xss_val)

    vlog = np.squeeze(vlog)
    predicted_centers = grid - xss_val

    r = np.sqrt(predicted_centers[:, :, 0] ** 2 + predicted_centers[:, :, 1] ** 2)
    theta = np.arctan2(predicted_centers[:, :, 0], predicted_centers[:, :, 1])

    predicted_centers_hough = np.stack([r, theta], -1)

    predicted_filter_mask, confidences = get_filtering_mask_by_segmentation_output(vlog, confidence_threshold=0.01)
    points = predicted_centers_hough[np.where(predicted_filter_mask)]

    clusters = dbscan_clustering(points, 0.3, 15)
    if len(clusters) == 0:
        print('No clusters!')
        return None

    clusters_filtered = [clu for clu in clusters if len(clu) > 200]
    clusters = clusters_filtered

    predicted_labels_points = vlog.argmax(2).astype(np.int32, copy=False)[np.where(predicted_filter_mask)]

    cluster_labels = []
    #cluster_confidence = np.random.rand(len(clusters))*0.5+0.5
    cluster_confidence=[]
    for cluster in clusters:
        cll = predicted_labels_points[cluster]
        #print(cll)
        ul = np.unique(cll)
        #print(ul)
        confs_cl=confidences.reshape((-1,))[cluster]

        c=[np.mean(confs_cl[cll==u]) for u in ul]

        freq = [len(cll[cll == u])/len(cll) for u in ul]

        c*=np.array(freq)
        cc=np.sum(c)
        cluster_confidence.append(cc)

        label = ul[np.argmax(freq)]
        cluster_labels.append(train_id_to_city_id[label])

    cluster_confidence=np.array(cluster_confidence)
    return points, clusters, cluster_labels, cluster_confidence, predicted_filter_mask


def dbscan_clustering(points,eps,npts):
    db = DBSCAN(eps=eps, min_samples=npts).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    clabels = db.labels_

    clusters = []
    for i in np.unique(clabels):
        if i == -1:
            continue
        clusters.append([k for k in np.where(clabels == i)[0]])

    return clusters




def main(argv=None):
    model = helper.import_module('model', FLAGS.model_path)
    train(model)


if __name__ == '__main__':
    tf.app.run()
