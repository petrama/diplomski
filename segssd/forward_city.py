

import numpy as np
import tensorflow as tf
import eval_helper
slim = tf.contrib.slim
import time

from PIL import Image

from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing

import city_tfrec_provider as dataset_provider

from nets import nets_factory
from tensorflow.contrib.slim.python.slim import queues

from nets import  segmentation


import os
#from notebooks import visualization
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_float(
    'weight_decay', 0.001, 'The weight decay on the model weights.')





net_shape = (300, 300)
data_format = 'NHWC'
## DATASET

#DATASET_DIR = '/home/pmarce/scp/city_bboxtfrec'
DATASET_DIR = '/home/pmarce/datasets/cityscapes/tfrecords'
SSD_CKPT='/home/pmarce/checkpoints/VGG_cityscapes_SSD_300x300_normal_iter_60000.ckpt'
SEG_CKPT='/home/pmarce/results/cityscapes/27_6_23-16-06/model.ckpt'

OUTPUT_DIR='/home/pmarce/datasets/cityscapes/segssd_results/'

SPLIT_NAME = 'val'
BATCH_SIZE = 1



det_classnames = ['person',
              'rider',
              'car',
              'truck',
              'bus',
              'train',
              'motorcycle',
              'bicycle',
              'trailer'
        ]

FLAGS = tf.app.flags.FLAGS

class_names=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation',
             'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

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






class_colors=[(128, 64, 128),#road
              (244, 35, 232),#side
                (70, 70, 70),#build
              (102, 102, 156),#wall
              (190, 153, 153),
              (153, 153, 153),
              (250, 170, 30),#fence,pole,traff light
              (220, 220, 0),#sign
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32)]

tf.app.flags.DEFINE_integer( 'img_height', 288,'')
tf.app.flags.DEFINE_integer( 'img_width', 640,'')
tf.app.flags.DEFINE_integer( 'num_channels', 3,'')
tf.app.flags.DEFINE_integer(
    'num_seg_classes', 19, 'Number of classes to use in the dataset.')

def process_image(rpredictions,rlocalisations,rbbox_img,ssd_anchors,select_threshold=0.5, nms_threshold=.45, net_shape=(300,300)):


    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=10, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes




def main(_):


    (image_gt, gbboxes, glabels, img_name, seg_labels) \
        = dataset_provider.inputs(DATASET_DIR, shuffle=False, num_epochs=False, dataset_partition=SPLIT_NAME)


    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        image_gt, glabels, gbboxes, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    image_seg_input=tf.expand_dims(tf.to_float(image_gt),0)

    with tf.variable_scope('model', reuse=False):
        seg_logits=segmentation.build(image_seg_input,None,is_training=False)

    ssd_class = nets_factory.get_network('ssd_300_vgg')
    ssd_params = ssd_class.default_params._replace(num_classes=10)
    ssd_net = ssd_vgg_300.SSDNet(ssd_params)
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

    isess=tf.InteractiveSession()
    # Restore SSD model.t
    #ckpt_filename = '/home/pmarce/scp/caffemodels/VGG_cityscapes_SSD_300x300_normal_iter_60000.ckpt'
    ckpt_filename=SSD_CKPT

    seg_ckpt_filename=SEG_CKPT



    ssd_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ssd_300_vgg')
    segmentation_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')


    isess.run(tf.global_variables_initializer())
    isess.run(tf.local_variables_initializer())
    saver_ssd = tf.train.Saver(ssd_vars)
    saver_ssd.restore(isess, ckpt_filename)

    saver_seg=tf.train.Saver(segmentation_vars)
    saver_seg.restore(isess,seg_ckpt_filename)



    start_time=time.time()
    with queues.QueueRunners(isess):
        # Start populating queues.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tp_fp_fn_global = {}

        for i in range(500):






            gtnp_image,  gtnp_labels, gtnp_boxes, gcl,gloc, rpred, rloc, rbbox,lpce,lnce,lloc,image_name ,segmentation_logits= isess.run(
                [image_gt,  glabels, gbboxes, gclasses,glocalisations,
                 predictions, localisations, bbox_img,loss_pce, loss_nce, loss_loc,img_name,seg_logits])

            im_name=image_name.decode('utf-8')



            #gt_h,gt_w,_=gtnp_shape
            pred_classes, pred_scores, pred_boxes = process_image(rpred, rloc, rbbox,ssd_anchors,select_threshold=0.01)
            predicted_labels = segmentation_logits.argmax(3).astype(np.int32, copy=False).reshape(FLAGS.img_height,
                                                                                                  FLAGS.img_width
                                                                                                  )


            gt=gtnp_image.copy()

            #visualization.bboxes_draw_on_img(gt,gtnp_boxes,thickness=1)
            output_dir=OUTPUT_DIR
            #plt.imsave('{}/{}_gt.png'.format(output_dir,im_name), gt)

            #plt.figure()
            #visualization.plt_bboxes(gtnp_image,pred_classes,pred_scores,pred_boxes,classnames=classnames)

            #fig=plt.gca()
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)

            #plt.savefig('eval_output/{}.png'.format(im_name),bbox_inches='tight',pad_inches=0)

            #evalstr = '%s/%s_segmented.png' % (output_dir, image_name)
            #print('seg_out_shape',segmentation_logits.shape)

            #eval_helper.draw_output(predicted_labels, class_colors, evalstr)

            town=im_name.split('_')[0]

            instances_output =os.path.join(output_dir, town)

            if not os.path.exists(instances_output):
                os.makedirs(instances_output)

            output_file = os.path.join(instances_output, im_name + '_outfile.txt')
            with open(output_file, 'w') as f:

                for i,(box,detection_label,dete_conf) in enumerate(zip(pred_boxes,pred_classes,pred_scores)):
                    ymin,xmin,ymax,xmax=box
                    ymin*=np.floor(FLAGS.img_height)
                    ymax*=np.ceil(FLAGS.img_height)
                    xmax*=np.ceil(FLAGS.img_width)
                    xmin*=np.floor(FLAGS.img_width)
                    #print('Raspon ',xmax-xmin,ymax-ymin)

                    indices_y=np.arange(ymin,ymax).astype(np.int32)
                    indices_x=np.arange(xmin,xmax).astype(np.int32)
                    indices=np.meshgrid(indices_y,indices_x)
                    #print(indices[0].shape,indices[1].shape)

                    segmentation_box=predicted_labels[indices]

                    #print(detection_label)
                    #print(det_classnames[detection_label-1])

                    det_name=det_classnames[detection_label-1]
                    if det_name=='trailer':
                        segmentation_label=255
                    else:

                        segmentation_label=class_names.index(det_name)
                    output_mask_box=np.zeros_like(segmentation_box)
                    output_mask_box [segmentation_box ==segmentation_label]=1

                    label_id=train_id_to_city_id[segmentation_label]

                    output_mask=np.zeros_like(predicted_labels)
                    output_mask[indices]=output_mask_box

                    if np.sum(output_mask)>200:



                        mask_file = os.path.join(instances_output, im_name + '_{}.png'.format(i))
                        #print(' '.join((name + '_{}.png'.format(i), str(clu_labels[i]), str(confidences[i]))), file=f)

                        #plt.imsave(evalstr,output_mask)
                        print(' '.join((im_name + '_{}.png'.format(i), str(label_id), str(dete_conf))), file=f)

                        im = Image.fromarray(output_mask.astype(np.uint16).astype(np.int32))
                        im.save(mask_file)


                    #print(box)
                    #print(segmentation_box.shape)
                    #break

                #break



            #for box,label,score in zip(pred_boxes,pred_classes,pred_scores):
                #print(' '.join([im_name,str(score),str(box[1]*gt_w),str(box[0]*gt_h),str(box[3]*gt_w),str(box[2]*gt_h)]),file=files[label-1])

        expired_time = time.time() - start_time

        print('Expired time: ', expired_time)







if __name__ == '__main__':
    tf.app.run()
