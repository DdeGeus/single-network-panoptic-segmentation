import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
resnet_arg_scope = resnet_utils.resnet_arg_scope

## Arguments in this file:
# regularization_weight
# batch_norm_decay
# batch_norm_istraining

def norm_arg_scope(params):
  batch_norm_params = {
    'decay': params.batch_norm_decay,
    'epsilon': 1e-5,
    'scale': True,
    'trainable': True,
    'is_training': params.batch_norm_istraining,
    'updates_collections': tf.GraphKeys.UPDATE_OPS}

  with tf.contrib.framework.arg_scope(
      [slim.conv2d],
      # weights_regularizer=slim.l2_regularizer(params.regularization_weight),
      weights_regularizer=None,
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with tf.contrib.framework.arg_scope(
        [slim.batch_norm],
        **batch_norm_params):
      with tf.contrib.framework.arg_scope(
          [slim.max_pool2d],
          padding='SAME') as arg_sc:
        return arg_sc

def feature_extractor(mode, features, params):

  with tf.variable_scope('feature_extractor'):

    # resnet base feature extractor scope arguments
    resnet_scope_args = {}
    if params.is_training:
      resnet_scope_args.update(weight_decay=params.regularization_weight,
                               batch_norm_decay=params.batch_norm_decay)

    # build base of feature extractor
    with tf.variable_scope('base'), (
            # slim.arg_scope(resnet_arg_scope(**resnet_scope_args))):
            slim.arg_scope(norm_arg_scope(params))):
      fe, end_points = resnet_v1.resnet_v1_50(
        features,
        num_classes=None,
        is_training=params.batch_norm_istraining,
        global_pool=False,
        output_stride=8)

    fe = slim.conv2d(fe, 512, [3, 3], scope="reduce_dims")

  return fe, end_points

def resnet_faster_rcnn_head(input, params):
  """
  Derived from https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow

  Args:
    input:
    params:

  Returns:

  """

  with tf.variable_scope('resnet_head', reuse=tf.AUTO_REUSE):
    block4 = [resnet_v1_block('block4', base_depth=256, num_units=3, stride=1)]

    with slim.arg_scope(norm_arg_scope(params)):
      C5, _ = resnet_v1.resnet_v1(input,
                                  block4,
                                  global_pool=False,
                                  include_root_block=False,
                                  scope='resnet_v1_50',
                                  reuse=tf.AUTO_REUSE)

      return C5