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

tf.app.flags.DEFINE_string('config_path', "config/cityscapes.py", """Path to experiment config.""")
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
        train_instance_mask, train_vector_centroid, train_img_name= reader.inputs(shuffle=True,
                                                                                  num_epochs=FLAGS.max_epochs,
                                                                                  dataset_partition='train')


        val_data, val_labels, val_instances, val_instance_mask,val_vector_centroid,val_img_name = reader.inputs(shuffle=False,
                                                                     num_epochs=FLAGS.max_epochs,
                                                                     dataset_partition='val')


        with tf.variable_scope('model'):
            logits, xss,loss,loss_ce,loss_reg, init_op, init_feed = model.build(train_data, train_labels,None,
                                                                     train_vector_centroid,train_instance_mask)
        with tf.variable_scope('model',reuse=True):
            logits_val,xss_val, loss_val,loss_ce_val,loss_reg_val = model.build(val_data, val_labels,None,
                                                                      val_vector_centroid, val_instance_mask,is_training=False)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('learning_rate', lr))

        opt = tf.train.AdamOptimizer(lr)

        grads = opt.compute_gradients(loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.add(tf.summary.histogram(var.op.name, var))

        # Add histograms for gradients.
        grad_tensors = []
        for grad, var in grads:
            grad_tensors += [grad]

            if grad is not None:
                summaries.add(tf.summary.histogram(var.op.name + '/gradients', grad))

        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_epochs,sharded=False)

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # Merge all summaries together.
        #summary_op = tf.summary.merge(list(summaries), name='summary_op')
        ## train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/events',
        #                                     sess.graph)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train',
                                               sess.graph)





        if len(FLAGS.resume_path) > 0:
            print('\nRestoring params from:', FLAGS.resume_path)


            init_fn=train_helper.get_init_fn(FLAGS)
            #saver.restore(sess, FLAGS.resume_path)
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            sess.run(tf.local_variables_initializer())


        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(init_op, feed_dict=init_feed)



        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)



        num_batches = FLAGS.train_size // FLAGS.batch_size

        plot_data = {}
        plot_data['train_loss'] = []
        plot_data['valid_loss'] = []
        plot_data['train_celoss'] = []
        plot_data['valid_celoss'] = []
        plot_data['train_regloss'] = []
        plot_data['valid_regloss'] = []
        plot_data['train_iou']=[]
        plot_data['valid_iou']=[]
        plot_data['train_acc'] = []
        plot_data['valid_acc'] = []
        plot_data['train_prec'] = []
        plot_data['valid_prec'] = []
        plot_data['train_rec'] = []
        plot_data['valid_rec'] = []
        plot_data['lr'] = []

        ex_start_time = time.time()

        global_step_value = 0

        visualize_dir = os.path.join(FLAGS.train_dir, 'visualize')

        for epoch_num in range(1, FLAGS.max_epochs + 1):
            print('\ntensorboard --logdir=' + FLAGS.train_dir + '\n')

            confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)

            avg_train_loss = 0
            avg_ce_loss=0
            avg_regr_loss=0


            for step in range(1, num_batches + 1):

                start_time = time.time()
                run_ops = [train_op, logits, loss, loss_ce, loss_reg,global_step, train_labels]

                if global_step_value % 50 == 0:
                    run_ops += [summary_op]
                    ret_val = sess.run(run_ops)
                    (_, logits_value, loss_value,loss_cv,loss_rv,global_step_value, batch_labels, summary_str) = ret_val
                    summary_writer.add_summary(summary_str, global_step_value)

                else:
                    ret_val = sess.run(run_ops)
                    (_, logits_value, loss_value,loss_cv,loss_rv,global_step_value, batch_labels) = ret_val

                duration = time.time() - start_time
                avg_train_loss += loss_value
                avg_regr_loss+=loss_rv
                avg_ce_loss+=loss_cv

                if step>=num_batches*4//5:
                    collect_confusion(logits_value,batch_labels,confusion_matrix)
                    acc,iou,rec,prec,size=eval_helper.compute_errors(confusion_matrix,'train',class_names,verbose=False)


                    if step  % 5 == 0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = '%s: epoch %d, batch %d / %d,\n' \
                                     'batch loss= %.2f\n' \
                                     'avg loss = %.2f\n' \
                                     'loss_class=%.2f\n' \
                                     'loss_reg=%.2f\n' \
                                     'avg_acc= %.2f\n' \
                                     'avg_iou= %.2f\n' \
                                     'avg prec=%.2f\n' \
                                     'avg rec=%.2f\n \
                                (%.1f examples/sec; %.3f sec/batch)'

                        print(format_str % (train_helper.get_expired_time(ex_start_time),
                                            epoch_num,
                                            step, num_batches + 1,
                                            loss_value,
                                            avg_train_loss / step,
                                            avg_ce_loss/step,
                                            avg_regr_loss/step,
                                            acc,
                                            iou,
                                            prec,
                                            rec,
                                            examples_per_sec, sec_per_batch))

                else:
                    if step%5==0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = '%s: epoch %d, batch %d / %d,\n' \
                                     'batch loss= %.2f\n'\
                                     'avg loss = %.2f\n' \
                                     'loss_class=%.2f\n' \
                                     'loss_reg=%.2f\n' \
                                     '(%.1f examples/sec; %.3f sec/batch)'

                        print(format_str % (train_helper.get_expired_time(ex_start_time),
                                            epoch_num,
                                            step, num_batches + 1,
                                            loss_value,
                                            avg_train_loss / step,
                                            avg_ce_loss / step,
                                            avg_regr_loss / step,
                                            examples_per_sec, sec_per_batch))



            train_loss = avg_train_loss / num_batches
            train_celoss=avg_ce_loss/num_batches
            train_regloss=avg_regr_loss/num_batches

            valid_acc,valid_iou, valid_prec, valid_rec, valid_loss,valid_clos,valid_rlos = evaluate(sess, epoch_num, val_labels, logits_val, loss_val,loss_ce_val,loss_reg_val,
                                                                    data_size=FLAGS.val_size,name='validation')

            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [valid_loss]

            plot_data['train_celoss'] += [train_celoss]
            plot_data['valid_celoss'] += [valid_clos]

            plot_data['train_regloss'] += [train_regloss]
            plot_data['valid_regloss'] += [valid_rlos]

            plot_data['train_iou']+=[iou]
            plot_data['valid_iou']+=[valid_iou]

            plot_data['train_acc'] += [acc]
            plot_data['valid_acc'] += [valid_acc]

            plot_data['train_prec'] += [prec]
            plot_data['valid_prec'] += [valid_prec]

            plot_data['train_rec'] += [rec]
            plot_data['valid_rec'] += [valid_rec]
            plot_data['lr'] += [lr.eval(session=sess)]

            eval_helper.plot_training_progress_new(visualize_dir, plot_data)

            if valid_iou >= max(plot_data['valid_iou']):
                print('Saving model...')
                t=time.time()
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path)
                print('Model is saved! t={}'.format(train_helper.get_expired_time(t)))



        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    model = helper.import_module('model', FLAGS.model_path)

    if tf.gfile.Exists(FLAGS.train_dir):
        raise ValueError('Train dir exists: ' + FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    stats_dir = os.path.join(FLAGS.train_dir, 'stats')
    tf.gfile.MakeDirs(stats_dir)
    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, 'visualize'))
    f = open(os.path.join(stats_dir, 'log.txt'), 'w')
    sys.stdout = train_helper.Logger(sys.stdout, f)

    copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
    copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))

    print('Experiment dir: ' + FLAGS.train_dir)
    train(model)


if __name__ == '__main__':
    tf.app.run()
