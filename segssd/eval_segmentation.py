import tensorflow as tf


import time
import os



import numpy as np
import tf_utils



from nets import nets_factory
import sys
from preprocessing import ssd_vgg_preprocessing
import city_tfrec_provider as dataset_provider
import helper
from shutil import copyfile
slim = tf.contrib.slim
import tensorflow.contrib.layers as layers

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_vgg', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 10, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer( 'img_height', 288,'')
tf.app.flags.DEFINE_integer( 'img_width', 640,'')
tf.app.flags.DEFINE_integer( 'num_channels', 3,'')

tf.app.flags.DEFINE_integer( 'num_seg_classes', 19,'')
tf.app.flags.DEFINE_integer( 'num_classes', 21,'')
tf.app.flags.DEFINE_integer('val_size',500,'')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'val', 'The name of the train/test split.')


tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')
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

def collect_confusion_matrix(y, yt, conf_mat,max_label):
  for i in range(y.size):
    l = y[i]
    lt = yt[i]
    if lt >= 0 and lt<max_label:
      conf_mat[l,lt] += 1

def compute_errors(conf_mat,name,class_names, verbose=True):
  num_correct = conf_mat.trace()
  num_classes = conf_mat.shape[0]
  total_size = conf_mat.sum()
  avg_pixel_acc = num_correct / total_size * 100.0
  TPFN = conf_mat.sum(0)
  TPFP = conf_mat.sum(1)
  FN = TPFN - conf_mat.diagonal()
  FP = TPFP - conf_mat.diagonal()
  class_iou = np.zeros(num_classes)
  class_recall = np.zeros(num_classes)
  class_precision = np.zeros(num_classes)
  if verbose:
    print(name + ' errors:')
  for i in range(num_classes):
    TP = conf_mat[i,i]
    if (TP + FP[i] + FN[i])>0:
      class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
    if TPFN[i] > 0:
      class_recall[i] = (TP / TPFN[i]) * 100.0
    else:
      class_recall[i] = 0
    if TPFP[i] > 0:
      class_precision[i] = (TP / TPFP[i]) * 100.0
    else:
      class_precision[i] = 0

    if verbose:
        print('\t%s IoU accuracy = %.2f %%' % (class_names[i], class_iou[i]))

  avg_class_iou = class_iou.mean()
  avg_class_recall = class_recall.mean()
  avg_class_precision = class_precision.mean()
  if verbose:
    print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
    print(name + ' mean class recall - TP / (TP+FN) = %.2f %%' % avg_class_recall)
    print(name + ' mean class precision - TP / (TP+FP) = %.2f %%' % avg_class_precision)
    print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
  return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size

def collect_confusion(logits, labels, conf_mat):
    predicted_labels = logits.argmax(3).astype(np.int32, copy=False)

    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    predicted_labels = np.resize(predicted_labels, [num_examples, ])
    batch_labels = np.resize(labels, [num_examples, ])

    assert predicted_labels.dtype == batch_labels.dtype
    collect_confusion_matrix(predicted_labels, batch_labels, conf_mat, FLAGS.num_seg_classes)



def evaluate(sess, epoch_num, labels, logits,  data_size,name):
    print('\nTest performance:')

    batch_size = FLAGS.batch_size
    print('testsize = ', data_size)
    assert data_size % batch_size == 0
    num_batches = data_size // batch_size
    start_time=time.time()

    conf_mat = np.zeros((FLAGS.num_seg_classes, FLAGS.num_seg_classes), dtype=np.uint64)
    for step in range(1,num_batches+1):
        if(step%5==0):
            print('evaluation: batch %d / %d '%(step,num_batches))

        out_logits,  batch_labels = sess.run([logits, labels])
        #continue


        collect_confusion(out_logits,batch_labels,conf_mat)


    print('Evaluation {} in epoch {} lasted {}'.format(name,epoch_num,helper.get_expired_time(start_time)))

    print('')

    (acc, iou, prec, rec, _) = compute_errors(conf_mat,name,class_names,verbose=True)


    print('IoU=%.2f Acc=%.2f Prec=%.2f Rec=%.2f' % (iou,acc, prec, rec))
    return acc,iou, prec, rec


def evaluate_model():
    with tf.Graph().as_default():
        # configure the training session
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        sess = tf.Session(config=config)


        (image_gt, gbboxes, glabels, img_name, seg_labels) \
            = dataset_provider.inputs(FLAGS.dataset_dir, shuffle=False, num_epochs=2, dataset_partition=FLAGS.dataset_split_name)

        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = \
            ssd_vgg_preprocessing.preprocess_for_train(image_gt, glabels, gbboxes,
                                                       out_shape=(300,300),
                                                       data_format='NHWC',
                                                       crop=False)



        # Training batches and queue.
        r = tf.train.batch(
            tf_utils.reshape_list([image, seg_labels]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)
        b_image,b_seg_labels = \
            tf_utils.reshape_list(r, [1,1])

        # Intermediate queueing: unique batch computation pipeline for all
        # GPUs running the training.
        batch_queue = slim.prefetch_queue.prefetch_queue(
            tf_utils.reshape_list([b_image,b_seg_labels]),
            capacity=2)

        b_image, b_seg_labels, = \
            tf_utils.reshape_list(batch_queue.dequeue(), [1,1])

        logits_val = model(b_image)




        saver = tf.train.Saver()

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

            if len(checkpoint_path) > 0:
                print('\nRestoring params from:', checkpoint_path)

                saver.restore(sess, checkpoint_path)
                sess.run(tf.local_variables_initializer())


        else:
            print('No resume path!')
            return



        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        valid_acc,valid_iou, valid_prec, valid_rec = evaluate(sess, 1, b_seg_labels, logits_val,

                                                                                        data_size=FLAGS.val_size,name='validation')



        coord.request_stop()
        coord.join(threads)
        sess.close()

def model(inputs):
    ssd_class = nets_factory.get_network(FLAGS.model_name)
    ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
    ssd_net = ssd_class(ssd_params)

    arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                  data_format='NHWC')
    with slim.arg_scope(arg_scope):

        with tf.variable_scope('ssd_300_vgg', 'ssd_300_vgg', [inputs], reuse=None):
            # Original VGG-16 blocks.
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3,3], scope='conv1')

            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # Block 2.
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # Block 3.
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')

            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # Block 4.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # Block 5.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],rate=1, scope='conv5')

        bn_params = {
            # Decay for the moving averages.
            'decay': 0.999,
            'center': True,
            'scale': True,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # None to force the updates
            'updates_collections': None,
            'is_training': False,

        }

    with tf.variable_scope('seg', reuse=None):
        with tf.contrib.framework.arg_scope([layers.convolution2d], stride=1, padding='SAME',
                                            weights_initializer=layers.variance_scaling_initializer(),
                                            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                            normalizer_params=bn_params,
                                            weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):
            net = layers.convolution2d(net, 512, kernel_size=7, scope='conv6_1', rate=4)
            net = layers.convolution2d(net, 512, kernel_size=3, scope='conv6_2', rate=8)

        logits = layers.convolution2d(net, FLAGS.num_seg_classes, 1, activation_fn=None, scope='unary_2')
        logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width], name='resize_score')

        return logits



def main(argv=None):
    evaluate_model()

if __name__ == '__main__':
    tf.app.run()
