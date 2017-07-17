import tensorflow as tf
import tensorflow.contrib.layers as layers


FLAGS = tf.app.flags.FLAGS

def build2(net, labels, weights=None, is_training=True):




    bn_params = {
        # Decay for the moving averages.
        'decay': 0.999,
        'center': True,
        'scale': True,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # None to force the updates
        'updates_collections': None,
        'is_training': is_training,

    }



    with tf.variable_scope('seg',reuse=None):

        with tf.contrib.framework.arg_scope([layers.convolution2d],stride=1,padding='SAME',
                                            weights_initializer=layers.variance_scaling_initializer(),
                                            activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
                                            normalizer_params=bn_params,
                                            weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):
            net = layers.convolution2d(net, 512, kernel_size=7, scope='conv6_1',rate=1)
            net = layers.convolution2d(net, 512, kernel_size=3, scope='conv6_2',rate=2)




        logits = layers.convolution2d(net, FLAGS.num_seg_classes, 1, activation_fn=None,scope='unary_2')

        logits=tf.image.resize_bilinear(logits,[FLAGS.img_height,FLAGS.img_width],name='resize_score')



    ce_loss=loss_pixel_classification(logits,labels,weights,is_training)
    return logits,tf.div(ce_loss,FLAGS.batch_size*FLAGS.img_height*FLAGS.img_width)



def build(inputs, labels, weights=None, is_training=True):



    weight_decay = 5e-4
    bn_params = {
        # Decay for the moving averages.
        'decay': 0.999,
        'center': True,
        'scale': True,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # None to force the updates
        'updates_collections': None,
        'is_training': is_training,

    }

    #with tf.variable_scope('ssd_300_vgg', 'ssd_300_vgg', [inputs], reuse=True):

    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
                                        # normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
                                        # weights_initializer=layers.variance_scaling_initializer(),
                                        normalizer_fn=None, weights_initializer=None):
                                        #weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, 64, scope='conv1_1')
        net = layers.convolution2d(net, 64, scope='conv1_2')
        net = layers.max_pool2d(net, 2, 2, scope='pool1')
        net = layers.convolution2d(net, 128, scope='conv2_1')
        net = layers.convolution2d(net, 128, scope='conv2_2')
        net = layers.max_pool2d(net, 2, 2, scope='pool2')
        net = layers.convolution2d(net, 256, scope='conv3_1')
        net = layers.convolution2d(net, 256, scope='conv3_2')
        net = layers.convolution2d(net, 256, scope='conv3_3')
        net = layers.max_pool2d(net, 2, 2, scope='pool3')

        net = layers.convolution2d(net, 512, scope='conv4_1')
        net = layers.convolution2d(net, 512, scope='conv4_2')
        net = layers.convolution2d(net, 512, scope='conv4_3')

        paddings = [[0, 0], [0, 0]]
        crops = [[0, 0], [0, 0]]

        block_size = 2

        net = tf.space_to_batch(net, paddings=paddings, block_size=block_size)
        net = layers.convolution2d(net, 512, scope='conv5_1')
        net = layers.convolution2d(net, 512, scope='conv5_2')
        net = layers.convolution2d(net, 512, scope='conv5_3')
        net = tf.batch_to_space(net, crops=crops, block_size=block_size)

    with tf.contrib.framework.arg_scope([layers.convolution2d], stride=1, padding='SAME',
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                        normalizer_params=bn_params,
                                        weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):
        net1 = layers.convolution2d(net, 512, kernel_size=7, scope='conv6_1', rate=4)
        net1 = layers.convolution2d(net1, 512, kernel_size=3, scope='conv6_2', rate=8)




    with tf.contrib.framework.arg_scope([layers.convolution2d], stride=1, padding='SAME',
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):
        logits = layers.convolution2d(net1, FLAGS.num_seg_classes, 1, padding='SAME', activation_fn=None, scope='unary_2')

    logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width], name='resize_score')


    return  logits


def loss_pixel_classification(logits,labels,weights,is_training):
    ce= weighted_cross_entropy_loss(logits, labels, weights)

    return ce



def weighted_cross_entropy_loss(logits, labels, weights=None, num_labels=1, max_weight=100):
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.op_scope([logits, labels], None, 'WeightedCrossEntropyLoss'):
    labels = tf.reshape(labels, shape=[num_examples])
    num_labels = tf.to_float(tf.reduce_sum(num_labels))
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_seg_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_seg_classes])
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_seg_classes])

    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(-tf.multiply(tf.to_float(one_hot_labels), log_softmax), 1)
    if weights != None:
      weights = tf.reshape(weights, shape=[num_examples])
      xent = tf.multiply(tf.minimum(tf.to_float(max_weight), weights), xent)
    total_loss = tf.div(tf.reduce_sum(xent), tf.to_float(num_labels), name='value')
    return tf.div(total_loss,tf.to_float(num_labels))

