import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class SemanticSegmentationModel(object):
  def __init__(self, params):
    self.params = params

  def predict(self, features):
    with tf.variable_scope('semantic_segmentation'):
      if self.params.psp_module:
        with tf.variable_scope('pyramid_pooling_module'):
          features = self._psp_module_original(features)

      logits = slim.conv2d(features, self.params.num_classes, kernel_size=1, activation_fn=None)

      if self.params.hybrid_upsampling:
        logits = slim.conv2d_transpose(
          inputs=logits,
          num_outputs=self.params.num_classes,
          kernel_size=3,
          padding='SAME',
          activation_fn=None,
          weights_initializer=slim.variance_scaling_initializer())
        logits = tf.image.resize_bilinear(logits, [self.params.height_input, self.params.width_input], align_corners=True)
      else:
        logits = tf.image.resize_bilinear(logits, [self.params.height_input, self.params.width_input], align_corners=True)

      return logits

  def loss(self, logits, labels, weights):
    with tf.variable_scope('semantic_segmentation_loss'):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      loss = tf.losses.compute_weighted_loss(loss, weights=weights)
      loss = tf.reduce_mean(loss)

      loss = loss * self.params.semantic_segmentation_loss_weight

    tf.summary.scalar('semantic', loss, family='losses')
    return loss

  def postprocess(self, logits):
    with tf.variable_scope('semantic_segmentation_postprocess'):
      probs = tf.nn.softmax(logits, axis=3)
      predictions = tf.argmax(probs, axis=3)

      return probs, predictions

  def _psp_module_original(self, features):
    pool1 = slim.layers.avg_pool2d(features, [60, 60], stride=[60, 60])
    conv1 = slim.conv2d(pool1, 512, 1, stride=1, activation_fn=None)
    bn1 = slim.batch_norm(conv1,
                          activation_fn=tf.nn.relu,
                          decay=0.9,
                          epsilon=1e-5,
                          is_training=self.params.batch_norm_istraining)
    ups1 = tf.image.resize_images(bn1, tf.shape(features)[1:3], align_corners=True)

    pool2 = slim.layers.avg_pool2d(features, [30, 30], stride=[30, 30])
    conv2 = slim.conv2d(pool2, 512, 1, stride=1, activation_fn=None)
    bn2 = slim.batch_norm(conv2,
                          activation_fn=tf.nn.relu,
                          decay=0.9,
                          epsilon=1e-5,
                          is_training=self.params.batch_norm_istraining)
    ups2 = tf.image.resize_images(bn2, tf.shape(features)[1:3], align_corners=True)

    pool3 = slim.layers.avg_pool2d(features, [20, 20], stride=[20, 20])
    conv3 = slim.conv2d(pool3, 512, 1, stride=1, activation_fn=None)
    bn3 = slim.batch_norm(conv3,
                          activation_fn=tf.nn.relu,
                          decay=0.9,
                          epsilon=1e-5,
                          is_training=self.params.batch_norm_istraining)
    ups3 = tf.image.resize_images(bn3, tf.shape(features)[1:3], align_corners=True)

    pool6 = slim.layers.avg_pool2d(features, [10, 10], stride=[10, 10])
    conv6 = slim.conv2d(pool6, 512, 1, stride=1, activation_fn=None)
    bn6 = slim.batch_norm(conv6,
                          activation_fn=tf.nn.relu,
                          decay=0.9,
                          epsilon=1e-5,
                          is_training=self.params.batch_norm_istraining)
    ups6 = tf.image.resize_images(bn6, tf.shape(features)[1:3], align_corners=True)

    concat = tf.concat([features, ups1, ups2, ups3, ups6], axis=-1)
    conv_final = slim.conv2d(concat, 512, 3, stride=1, activation_fn=None)
    bn_final = slim.batch_norm(conv_final,
                          activation_fn=tf.nn.relu,
                          decay=0.9,
                          epsilon=1e-5,
                          is_training=self.params.batch_norm_istraining)

    return bn_final

  def format_gt(self, labels_in):
    labels = tf.cast(labels_in, tf.int32)
    labels = tf.gather(tf.cast(self.params.labels2cids, tf.int32), labels)
    weights = tf.cast(tf.greater_equal(labels, 0), tf.int32)
    labels = tf.multiply(labels, weights)

    return labels, weights
