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
from utils import box_utils

class BoxMatcher(object):

  def __init__(self,
               iou_upper_th=0.7,
               iou_lower_th=0.3,
               unmatched_cls_target=None,
               encoder_scales=None,
               return_iids=False,
               unmatched_iid_target=None):

    self.iou_upper_th = iou_upper_th
    self.iou_lower_th = iou_lower_th

    self.encoder_scales = encoder_scales

    if unmatched_cls_target is None:
      self._unmatched_cls_target = [0.]
    else:
      self._unmatched_cls_target = unmatched_cls_target

    if unmatched_iid_target is None:
      self._unmatched_iid_target = [-1.]
    else:
      self._unmatched_iid_target = unmatched_iid_target

    self.return_iids = return_iids

  def match(self, gt_boxes, det_boxes, gt_labels=None, gt_weights=None, gt_iid_labels=None):

    with tf.variable_scope("MatchBoxes"):

      if gt_labels is None:
        gt_labels = tf.ones(tf.expand_dims(tf.shape(gt_boxes)[0], 0))
        gt_labels = tf.expand_dims(gt_labels, -1)

      if gt_weights is None:
        gt_weights = tf.ones(tf.expand_dims(tf.shape(gt_boxes)[0], 0))
        gt_weights = tf.expand_dims(gt_weights, -1)

      if gt_iid_labels is None:
        self.return_iids = False

      gt_weights = tf.cast(gt_weights, tf.float32)

      ious = box_utils.calculate_ious(gt_boxes, det_boxes)

      max_ious = tf.reduce_max(ious, 0)
      matches = tf.argmax(ious, 0, output_type=tf.int32)

      below_lower_threshold = tf.greater(self.iou_lower_th, max_ious)
      between_thresholds = tf.logical_and(
        tf.greater_equal(max_ious, self.iou_lower_th),
        tf.greater(self.iou_upper_th, max_ious))

      matches = _set_values_using_indicator(matches,
                                            below_lower_threshold,
                                            -1)
      matches = _set_values_using_indicator(matches,
                                            between_thresholds,
                                            -2)

      box_targets = self.get_box_targets(matches, det_boxes, gt_boxes)
      cls_targets = self.get_cls_targets(matches, gt_labels)
      box_weights = self.get_box_weights(matches, gt_weights)
      cls_weights = self.get_cls_weights(matches, gt_weights)

      if self.return_iids:
        iid_targets = self.get_iid_targets(matches, gt_iid_labels)
        return box_targets, cls_targets, box_weights, cls_weights, iid_targets
      else:
        return box_targets, cls_targets, box_weights, cls_weights

  def get_box_targets(self, matches, det_boxes, gt_boxes):
    ignored_value = tf.zeros(4)
    unmatched_value = tf.zeros(4)
    default_regression_target = tf.constant([4 * [0]], tf.float32)

    # Ignored and unmatched (value -2 and -1, respectively) get zeros as coordinates to gather from
    gt_boxes_padded = tf.concat([tf.stack([ignored_value, unmatched_value]),
                                 gt_boxes], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    matched_gt_boxes = tf.gather(gt_boxes_padded, gather_indices)

    matched_box_targets = box_utils.encode_boxes(matched_gt_boxes,
                                                 det_boxes,
                                                 scale_factors=self.encoder_scales)

    # Zero out the unmatched and ignored regression targets.
    unmatched_ignored_reg_targets = tf.tile(
      default_regression_target, [tf.shape(matched_gt_boxes)[0], 1])
    matched_anchors_mask = tf.greater_equal(matches, 0)
    box_targets = tf.where(matched_anchors_mask,
                           matched_box_targets,
                           unmatched_ignored_reg_targets)

    return box_targets

  def get_cls_targets(self, matches, gt_labels):
    ignored_value = self._unmatched_cls_target
    unmatched_value = self._unmatched_cls_target
    input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),
                              gt_labels], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    cls_targets = tf.gather(input_tensor, gather_indices)

    return cls_targets

  def get_box_weights(self, matches, gt_weights):
    ignored_value = 0.
    unmatched_value = 0.
    input_tensor = tf.concat([tf.stack([[ignored_value], [unmatched_value]]),
                              gt_weights], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    reg_weights = tf.gather(input_tensor, gather_indices)

    return reg_weights

  def get_cls_weights(self, matches, gt_weights):
    ignored_value = 0.
    unmatched_value = 1.
    input_tensor = tf.concat([tf.stack([[ignored_value], [unmatched_value]]),
                              gt_weights], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    cls_weights = tf.gather(input_tensor, gather_indices)

    return cls_weights

  def get_iid_targets(self, matches, gt_labels):
    ignored_value = self._unmatched_iid_target
    unmatched_value = self._unmatched_iid_target
    input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),
                              gt_labels], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    iid_targets = tf.gather(input_tensor, gather_indices)

    return iid_targets


def batch_match_boxes(box_matcher, gt_box_wrappers, det_boxes):
  box_targets_list = list()
  cls_targets_list = list()
  box_weights_list = list()
  cls_weights_list = list()

  for gt_box_wrapper in gt_box_wrappers:
    gt_boxes = gt_box_wrapper.get_boxes()
    gt_weights = gt_box_wrapper.get_weights()

    box_targets, cls_targets, box_weights, cls_weights = box_matcher.match(gt_boxes,
                                                                           det_boxes,
                                                                           gt_weights=gt_weights)

    box_targets_list.append(box_targets)
    cls_targets_list.append(cls_targets)
    box_weights_list.append(box_weights)
    cls_weights_list.append(cls_weights)

  return (tf.stack(box_targets_list),
          tf.stack(cls_targets_list),
          tf.stack(box_weights_list),
          tf.stack(cls_weights_list))

def _set_values_using_indicator(x, indicator, val):
  """Set the indicated fields of x to val.

  Args:
    x: tensor.
    indicator: boolean with same shape as x.
    val: scalar with value to set.

  Returns:
    modified tensor.
  """
  indicator = tf.cast(indicator, x.dtype)
  return tf.add(tf.multiply(x, 1 - indicator), val * indicator)