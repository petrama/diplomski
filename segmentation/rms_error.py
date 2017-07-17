import tensorflow as tf
import read_cityscapes_tf_records as reader
import train_helper
import time
import os


import eval_helper
import numpy as np

import helper

import sys

from shutil import copyfile

tf.app.flags.DEFINE_string('config_path', "config/rmse.py", """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.__dict__['__flags'].keys())



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

def collect_confusion(logits, labels, conf_mat):
    predicted_labels = logits.argmax(3).astype(np.int32, copy=False)

    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    predicted_labels = np.resize(predicted_labels, [num_examples, ])
    batch_labels = np.resize(labels, [num_examples, ])

    assert predicted_labels.dtype == batch_labels.dtype
    eval_helper.collect_confusion_matrix(predicted_labels, batch_labels, conf_mat, FLAGS.num_classes)



def evaluate(sess, epoch_num, labels, logits, loss, loss_ce,loss_reg, data_size,name):
    print('\nTest performance:')
    loss_avg = 0
    loss_ce_avg=0
    loss_reg_avg=0

    batch_size = FLAGS.batch_size
    print('testsize = ', data_size)
    assert data_size % batch_size == 0
    num_batches = data_size // batch_size
    start_time=time.time()

    conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
    for step in range(1,num_batches+1):
        if(step%5==0):
            print('evaluation: batch %d / %d '%(step,num_batches))

        out_logits, loss_value,loss_ce_value,loss_reg_value, batch_labels = sess.run([logits, loss,loss_ce,loss_reg, labels])

        loss_avg += loss_value
        loss_ce_avg+=loss_ce_value
        loss_reg_avg+=loss_reg_value

        collect_confusion(out_logits,batch_labels,conf_mat)


    print('Evaluation {} in epoch {} lasted {}'.format(name,epoch_num,train_helper.get_expired_time(start_time)))

    print('')

    (acc, iou, prec, rec, _) = eval_helper.compute_errors(conf_mat,name,class_names,verbose=True)
    loss_avg /= num_batches;
    loss_reg_avg/=num_batches;
    loss_ce_avg/=num_batches;

    print('IoU=%.2f Acc=%.2f Prec=%.2f Rec=%.2f' % (iou,acc, prec, rec))
    return acc,iou, prec, rec, loss_avg,loss_ce_avg,loss_reg_avg



def frmse(gt,predicted):
    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    gtr = tf.reshape(gt, (num_examples, 2))
    pred = tf.reshape(predicted, (num_examples, 2))
    diff = tf.pow(gtr - pred,2)
    return tf.reduce_sum(diff[:,0]),tf.reduce_sum(diff[:,1])


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

        train_data, train_labels, train_instances,\
        train_instance_mask, train_vector_centroid, train_img_name= reader.inputs(shuffle=False,
                                                                                  num_epochs=1,
                                                                                  dataset_partition='train')


        val_data, val_labels, val_instances, val_instance_mask,val_vector_centroid,val_img_name = reader.inputs(shuffle=False,
                                                                     num_epochs=1,
                                                                     dataset_partition='val')


        with tf.variable_scope('model'):
            logits,xss, loss,loss_ce,loss_reg= model.build(train_data, train_labels,None,
                                                                     train_vector_centroid,train_instance_mask,is_training=False)
        with tf.variable_scope('model',reuse=True):
            logits_val,xss_val, loss_val,loss_ce_val,loss_reg_val = model.build(val_data, val_labels,None,
                                                                      val_vector_centroid, val_instance_mask,is_training=False)


        rmse_batch_y,rmse_batch_x=frmse(train_vector_centroid,xss)
        rmse_val_batch_y,rmse_val_batch_x=frmse(val_vector_centroid,xss_val)

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

        num_batches = FLAGS.val_size // FLAGS.batch_size

        rmse_x = 0
        rmse_y = 0

        for step in range(1, num_batches + 1):
            # print(step)
            run_ops = rmse_val_batch_y, rmse_val_batch_x
            ret_val = sess.run(run_ops)
            val_rmse_y, val_rmse_x = ret_val
            rmse_y += val_rmse_y
            rmse_x += val_rmse_x

        nn = FLAGS.val_size * FLAGS.img_height * FLAGS.img_width
        print('Val mse x', np.sqrt(np.array(rmse_x) / np.array(nn)))
        print('Val mse y', np.sqrt(np.array(rmse_y) / np.array(nn)))

        num_batches = FLAGS.train_size // FLAGS.batch_size

        rmse_x=0
        rmse_y=0

        for step in range(1, num_batches + 1):
            #print(step)
            run_ops = rmse_batch_y,rmse_batch_x
            ret_val = sess.run(run_ops)
            val_rmse_y ,val_rmse_x= ret_val
            rmse_x+=val_rmse_x
            rmse_y+=val_rmse_y

        nn = FLAGS.train_size * FLAGS.img_height * FLAGS.img_width
        print('Train  rmse x',np.sqrt(np.array(rmse_x) / np.array(nn)))
        print('Train  rmse y', np.sqrt(np.array(rmse_y) / np.array(nn)))





        coord.request_stop()
        coord.join(threads)
        sess.close()





def main(argv=None):
    model = helper.import_module('model', FLAGS.model_path)
    train(model)


if __name__ == '__main__':
    tf.app.run()
