# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Tensorflow Object Detection API code adapted by Daan de Geus

import tensorflow as tf

def rpn_regression_loss(prediction_tensor, target_tensor, weights):
  """Compute loss function.

  Args:
    prediction_tensor: A float tensor of shape [batch_size, num_anchors,
      code_size] representing the (encoded) predicted locations of objects.
    target_tensor: A float tensor of shape [batch_size, num_anchors,
      code_size] representing the regression targets
    weights: a float tensor of shape [batch_size, num_anchors]

  Returns:
    loss: a float tensor of shape [batch_size, num_anchors] tensor
      representing the value of the loss function.
  """
  with tf.variable_scope("RPNRegressionLoss"):
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    anchorwise_smooth_l1norm = tf.reduce_sum(
      tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
      2) * weights
    return anchorwise_smooth_l1norm

def rpn_objectness_loss(prediction_tensor, target_tensor, weights):
  """Compute loss function.

  Args:
    prediction_tensor: A float tensor of shape [batch_size, num_anchors,
      num_classes] representing the predicted logits for each class
    target_tensor: A float tensor of shape [batch_size, num_anchors,
      num_classes] representing one-hot encoded classification targets
    weights: a float tensor of shape [batch_size, num_anchors]

  Returns:
    loss: a float tensor of shape [batch_size, num_anchors]
      representing the value of the loss function.
  """
  with tf.variable_scope("RPNObjectnessLoss"):
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    prediction_tensor = tf.divide(
        prediction_tensor, 1.0, name='scale_logit')
    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights

def detection_regression_loss(prediction_tensor, target_tensor, weights):
  """Compute loss function.

  Args:
    prediction_tensor: A float tensor of shape [batch_size, num_anchors,
      code_size] representing the (encoded) predicted locations of objects.
    target_tensor: A float tensor of shape [batch_size, num_anchors,
      code_size] representing the regression targets
    weights: a float tensor of shape [batch_size, num_anchors]

  Returns:
    loss: a float tensor of shape [batch_size, num_anchors] tensor
      representing the value of the loss function.
  """
  with tf.variable_scope("DetectionRegressionLoss"):
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    anchorwise_smooth_l1norm = tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        2) * weights
    return anchorwise_smooth_l1norm

def detection_classification_loss(prediction_tensor, target_tensor, weights):
  """Compute loss function.

  Args:
    prediction_tensor: A float tensor of shape [batch_size, num_anchors,
      num_classes] representing the predicted logits for each class
    target_tensor: A float tensor of shape [batch_size, num_anchors,
      num_classes] representing one-hot encoded classification targets
    weights: a float tensor of shape [batch_size, num_anchors]

  Returns:
    loss: a float tensor of shape [batch_size, num_anchors]
      representing the value of the loss function.
  """
  with tf.variable_scope("DetectionClassificationLoss"):
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    prediction_tensor = tf.divide(
        prediction_tensor, 1.0, name='scale_logit')
    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights