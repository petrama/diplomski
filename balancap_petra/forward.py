
import argparse
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing


from datasets import pascalvoc_2007

from tensorflow.contrib.slim.python.slim import queues

import os



net_shape = (300, 300)
data_format = 'NHWC'
## DATASET

DATASET_DIR = '/home/pmarce/datasets/VOCdevkit/VOC2007/test_tfrecords'
SPLIT_NAME = 'test'
BATCH_SIZE = 1

classnames = ['aeroplane',
              'bicycle',
              'bird',
              'boat',
              'bottle',
              'bus',
              'car',
              'cat',
              'chair',
              'cow',
              'diningtable',
              'dog',
              'horse',
              'motorbike',
              'person',
              'pottedplant',
              'sheep',
              'sofa',
              'train',
              'tvmonitor']

def process_image(rpredictions,rlocalisations,rbbox_img,ssd_anchors,select_threshold=0.01, nms_threshold=.45, net_shape=(300,300)):


    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes




def main(ckpt_filename):
    labels = sorted(pascalvoc_2007.TRAIN_STATISTICS.keys())
    labels.remove('total')
    labels.remove('none')

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)


    dataset = pascalvoc_2007.get_split(SPLIT_NAME, DATASET_DIR)
    data_sources = dataset.data_sources
    # print(data_sources)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              shuffle=False,
                                                              #                                                           num_epochs=1,
                                                              common_queue_capacity=2 * BATCH_SIZE,
                                                              common_queue_min=BATCH_SIZE)

    [image_raw, shape_raw, glabels_raw, gbboxes_raw,name] = provider.get(['image', 'shape',
                                                                     'object/label',
                                                                     'object/bbox','name'])


    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        image_raw, glabels_raw, gbboxes_raw, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)
    ssd_net = ssd_vgg_300.SSDNet()
    ssd_anchors = ssd_net.anchors(net_shape)



    gclasses, glocalisations, gscores = \
        ssd_net.bboxes_encode(labels_pre, bboxes_pre, ssd_anchors)



    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    #with slim.arg_scope(ssd_net.arg_scope_caffe(caffemodel)):
        predictions, localisations, logits, _ = ssd_net.net(image_4d, is_training=False, reuse=False)

    loss_pce, loss_nce, loss_loc = ssd_net.losses(logits, localisations,
                                                  gclasses, glocalisations, gscores,
                                                  match_threshold=0.5,
                                                  negative_ratio=3,
                                                  alpha=1.0,
                                                  label_smoothing=0.0)

    # Restore SSD model.
    #ckpt_filename = '/home/pmarce/caffemodels/VGG_VOC0712_SSD_300x300_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    isess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)
    save_figs = False
    output='./eval_output'

    files = [
        open(
            os.path.join(
            output, 'comp4_det_test_{:s}.txt'.format(label)),
            mode='w')
        for label in labels]

    print(len(data_sources))
    

    with queues.QueueRunners(isess):
        # Start populating queues.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tp_fp_fn_global = {}

        for i,k in enumerate(data_sources):



            print('Data source',k)



            gtnp_image, gtnp_shape, gtnp_labels, gtnp_boxes, gcl,gloc, rpred, rloc, rbbox,lpce,lnce,lloc,image_name = isess.run(
                [image_raw, shape_raw, glabels_raw, gbboxes_raw, gclasses,glocalisations,
                 predictions, localisations, bbox_img,loss_pce, loss_nce, loss_loc,name])

            im_name=image_name.decode('utf-8')

            gt_h,gt_w,_=gtnp_shape
            pred_classes, pred_scores, pred_boxes = process_image(rpred, rloc, rbbox,ssd_anchors)


            for box,label,score in zip(pred_boxes,pred_classes,pred_scores):
                print(' '.join([im_name,str(score),str(box[1]*gt_w),str(box[0]*gt_h),str(box[3]*gt_w),str(box[2]*gt_h)]),file=files[label-1])

    for f in files:
        f.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='/home/pmarce/caffemodels/VGG_VOC0712_SSD_300x300_iter_120000.ckpt')
    main(parser.parse_args().model)
