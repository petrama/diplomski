import tensorflow as tf
import tensorflow.contrib.layers as layers
from model_helper import read_vgg_init


import losses

FLAGS = tf.app.flags.FLAGS


def total_loss_sum(losses):
    # Assemble all of the losses for the current tower only.
    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    return total_loss


def create_init_op(vgg_layers):
    variables = tf.contrib.framework.get_variables()
    init_map = {}
    for var in variables:
        name_split = var.name.split('/')
        if len(name_split) != 3:
            continue
        name = name_split[1] + '/' + name_split[2][:-2]
        if name in vgg_layers:
            print(var.name, ' --> init from ', name)
            init_map[var.name] = vgg_layers[name]
            print(var.name,vgg_layers[name].shape)
        else:
            print(var.name, ' --> random init')



    init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
    return init_op, init_feed

def pyramid_pooling_layer(net,subsample_factor):
    sd = net.get_shape().as_list()[1:3]
    sd1 = [sd[0], sd[1]]
    sd2 = [sd[0]// 2, sd[1]// 2]
    sd3 = [sd1[0] // 3, sd1[1] // 3]
    sd4 = [sd1[0] // 6, sd1[1] // 6]
    upsampled_size=[FLAGS.img_height//subsample_factor, FLAGS.img_width//subsample_factor]

    first = layers.avg_pool2d(net, kernel_size=sd1)
    first_conv = layers.convolution2d(first, 128, kernel_size=1)
    first_up = tf.image.resize_bilinear(first_conv,upsampled_size , name='spp-1')

    second = layers.max_pool2d(net, kernel_size=sd2, stride=sd2)
    second_conv = layers.convolution2d(second, 128, kernel_size=1, scope='spp-2')
    second_up = tf.image.resize_bilinear(second_conv, upsampled_size, name='spp-2')

    third = layers.max_pool2d(net, kernel_size=sd3, stride=sd3)
    third_conv = layers.convolution2d(third, 128, kernel_size=1, scope='spp-3')
    third_up = tf.image.resize_bilinear(third_conv, upsampled_size, name='spp-3')

    forth = layers.max_pool2d(net, kernel_size=sd4, stride=sd4)
    forth_conv = layers.convolution2d(forth, 128, kernel_size=1, scope='spp-4')
    forth_up = tf.image.resize_bilinear(forth_conv,upsampled_size, name='spp-4')

    stacked=tf.concat([first_up,second_up,third_up,forth_up],axis=3,name='spp_global_context')
    stacked=tf.concat([net,stacked],axis=3,name='spp')




    print('result shape',stacked.get_shape())
    return stacked

def build(inputs, labels, weights,vector_centr,instance_mask, is_training=True):

    vgg_layers, vgg_layer_names = read_vgg_init(FLAGS.vgg_init_dir)


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
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
                                        # normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
                                        # weights_initializer=layers.variance_scaling_initializer(),
                                        normalizer_fn=None, weights_initializer=None,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
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

        net=tf.space_to_batch(net,paddings=paddings,block_size=block_size)
        net = layers.convolution2d(net, 512, scope='conv5_1')
        net = layers.convolution2d(net, 512, scope='conv5_2')
        net = layers.convolution2d(net, 512, scope='conv5_3')
        net=tf.batch_to_space(net,crops=crops,block_size=block_size)



        #net=pyramid_pooling_layer(net,8)


    with tf.contrib.framework.arg_scope([layers.convolution2d],stride=1,padding='SAME',
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
                                        normalizer_params=bn_params,
                                        weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):
        net1 = layers.convolution2d(net, 512, kernel_size=7, scope='conv6_1',rate=4)
        net1 = layers.convolution2d(net1, 512, kernel_size=3, scope='conv6_2',rate=8)

    with tf.contrib.framework.arg_scope([layers.convolution2d], stride=1, padding='SAME',
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                        normalizer_params=bn_params,
                                        weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):


        with tf.variable_scope('centers'):
            xss = layers.convolution2d(net, 512, kernel_size=7, scope='conv6_1', rate=4)
            xss = layers.convolution2d(xss, 512, kernel_size=3, scope='conv6_2', rate=2)
            xss = layers.convolution2d(xss, 512, kernel_size=3, scope='conv6_3', rate=2)

    with tf.contrib.framework.arg_scope([layers.convolution2d],stride=1,padding='SAME',
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):

        logits = layers.convolution2d(net1, FLAGS.num_classes, 1,padding='SAME', activation_fn=None,scope='unary_2')
        xss=layers.convolution2d(xss, 2, 1 ,padding='SAME', activation_fn=None,scope='centroid_regression')



    logits=tf.image.resize_bilinear(logits,[FLAGS.img_height,FLAGS.img_width],name='resize_score')
    xss = tf.image.resize_bilinear(xss, [FLAGS.img_height, FLAGS.img_width], name='vector_to_centroid')


    log_sigma_class=tf.Variable(initial_value=3.5,trainable=True)
    log_sigma_regr=tf.Variable(initial_value=3.5,trainable=True)


    regr_loss = loss_object_centorid_regression(vector_centr, xss,is_training)
    ce_loss=loss_pixel_classification(logits,labels,weights,is_training)

    loss=get_loss(ce_loss,regr_loss,log_sigma_class,log_sigma_regr,is_training)


    if is_training:
        init_op, init_feed = create_init_op(vgg_layers)
        return logits, loss,ce_loss,regr_loss, init_op, init_feed

    return logits,loss,ce_loss,regr_loss




def rmse(gt,predicted,instance_mask):
    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    gtr = tf.reshape(gt, (num_examples, 2))
    pred = tf.reshape(predicted, (num_examples, 2))
    diff = tf.pow(gtr - pred,2)
    return tf.reduce_sum(diff),tf.reduce_sum(instance_mask)



def loss_object_centorid_regression(gt,predicted,is_training):
    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    gtr=tf.reshape(gt,(num_examples,2))
    pred=tf.reshape(predicted,(num_examples,2))
    diff=tf.abs(gtr-pred)
    norm=tf.reduce_sum(diff, axis=1)
    #l1loss= tf.div(tf.reduce_sum(norm),num_examples)
    l1loss=tf.reduce_sum(norm)
    if is_training:
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('centroid regr loss',l1loss))

    return l1loss


def loss_pixel_classification(logits,labels,weights,is_training):
    ce= losses.weighted_cross_entropy_loss(logits, labels, weights)
    if is_training:
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('pixel class loss', ce))

    return ce




def get_loss(class_loss,regr_loss, log_sigma_class,log_sigma_regr,is_training):


    total_loss = total_loss_sum([class_loss/(2*tf.exp(log_sigma_class))+log_sigma_class,
                                 regr_loss/(2*tf.exp(log_sigma_regr))+log_sigma_regr])
    if is_training:
        loss_averages_op = losses.add_loss_summaries(total_loss)
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

    return total_loss
