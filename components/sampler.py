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
from utils.utils import indices_to_dense_vector

class RPNSampler(object):

  def __init__(self, positive_rate=0.5):
    self.positive_rate = positive_rate

  def sample(self, indicator, minibatch_size, labels):
    """

    Args:
      indicator: indicator indicating what entries can be sampled [N] (tf.bool)
      minibatch_size: the amount of entries to be sampled [] (integer)
      labels: the label for all tensors [N] (tf.bool)

    Returns: indicator for sampled entries [N] (tf.bool)

    """

    with tf.variable_scope("Sample"):

      # Only sample from indicated samples
      neg_samples = tf.logical_not(labels)
      neg_samples = tf.logical_and(neg_samples, indicator)
      pos_samples = tf.logical_and(labels, indicator)

      # Find number of positive entries to be sampled and sample
      max_pos_samples = int(self.positive_rate * minibatch_size)
      num_pos_samples = tf.minimum(max_pos_samples, tf.reduce_sum(tf.cast(pos_samples, tf.int32)))
      pos_sampled = self.sample_per_indicator(pos_samples, num_pos_samples)

      # Find number of negative entries to be sampled and sample
      num_neg_samples = minibatch_size - num_pos_samples
      neg_sampled = self.sample_per_indicator(neg_samples, num_neg_samples)

      # Combine entries to be sampled
      sampled = tf.logical_or(pos_sampled, neg_sampled)

    return sampled

  @staticmethod
  def sample_per_indicator(indicator, num_samples):

    indices = tf.where(indicator)
    indices = tf.random_shuffle(indices)

    sampled = indices[0:num_samples]

    samples_dense = indices_to_dense_vector(sampled, tf.shape(indicator)[0])
    samples_dense = tf.cast(samples_dense, tf.bool)

    return samples_dense


class ROISampler(object):

  def __init__(self, params, positive_rate=0.25):
    self.positive_rate = positive_rate
    self.sampler = RPNSampler(positive_rate=positive_rate)
    self.params = params

  def batch_sample_and_match(self, rois_batch, gt_box_wrappers, minibatch_size, box_matcher):
    """

    Args:
      rois_batch: Bbox coordinates for regions of interest [Nb, RPN_minibatch_size, 4]
      gt_box_wrappers: list of BoxWrappers [Nb]
      minibatch_size: Number of ROIs to be sampled for detection branch
      box_matcher: The box matcher function to match the ROIs to the ground truth

    Returns:
      rois_sampled_batch: Bbox coordinates for the sampled ROIS [Nb, RPN_minibatch_size, 4]
      matches: A dictionary with the batched box, cls targets and box, cls weights

    """
    with tf.variable_scope("BatchSampleAndMatch"):

      rois_batch = tf.unstack(rois_batch, num=self.params.Nb)

      matches = dict()

      rois_sampled_batch = list()
      box_targets_sampled_batch = list()
      cls_targets_sampled_batch = list()
      box_weights_sampled_batch = list()
      cls_weights_sampled_batch = list()
      iid_targets_sampled_batch = list()

      for rois, gt_box_wrapper in zip(rois_batch, gt_box_wrappers):
        gt_boxes = gt_box_wrapper.get_boxes()
        gt_classes = gt_box_wrapper.get_classes()
        gt_classes_with_background = self.format_box_gt(gt_classes)
        gt_classes_with_background = tf.reshape(gt_classes_with_background,
                                                [-1, (self.params.num_things_classes+1)])
        gt_weights = gt_box_wrapper.get_weights()
        gt_instance_ids = gt_box_wrapper.get_instance_ids()

        # Match ROIs to the ground truth and get loss targets and weights
        box_targets, cls_targets, box_weights, cls_weights, iid_targets = box_matcher.match(gt_boxes,
                                                                                            rois,
                                                                                            gt_labels=gt_classes_with_background,
                                                                                            gt_weights=gt_weights,
                                                                                            gt_iid_labels=gt_instance_ids)

        # Sample the ROIs based on the classification weights and targets
        indicator = tf.cast(tf.squeeze(cls_weights, 1), tf.bool)
        positive_indicator = tf.greater(tf.argmax(cls_targets, axis=1), 0)
        sampled_ids = self.sampler.sample(indicator, minibatch_size, positive_indicator)

        # Gather the sampled targets and weights
        box_targets_sampled = tf.boolean_mask(box_targets, sampled_ids)
        cls_targets_sampled = tf.boolean_mask(cls_targets, sampled_ids)
        box_weights_sampled = tf.boolean_mask(box_weights, sampled_ids)
        cls_weights_sampled = tf.boolean_mask(cls_weights, sampled_ids)
        iid_targets_sampled = tf.boolean_mask(iid_targets, sampled_ids)

        # iid_targets_sampled = tf.Print(iid_targets_sampled, [iid_targets_sampled], summarize=20, message='iid_targets_sampled')
        # cls_targets_sampled = tf.Print(cls_targets_sampled, [cls_targets_sampled], summarize=20, message='cls_targets_sampled')

        box_targets_sampled_batch.append(box_targets_sampled)
        cls_targets_sampled_batch.append(cls_targets_sampled)
        box_weights_sampled_batch.append(box_weights_sampled)
        cls_weights_sampled_batch.append(cls_weights_sampled)
        iid_targets_sampled_batch.append(iid_targets_sampled)

        # Gather the sampled ROIs
        rois_sampled = tf.boolean_mask(rois, sampled_ids)
        rois_sampled_batch.append(rois_sampled)

      matches['box_targets'] = tf.stack(box_targets_sampled_batch)
      matches['cls_targets'] = tf.stack(cls_targets_sampled_batch)
      matches['box_weights'] = tf.stack(box_weights_sampled_batch)
      matches['cls_weights'] = tf.stack(cls_weights_sampled_batch)
      matches['iid_targets'] = tf.stack(iid_targets_sampled_batch)

      rois_sampled_batch = tf.stack(rois_sampled_batch)

    return rois_sampled_batch, matches


  def format_box_gt(self, gt_classes):
    # gt_classes = tf.Print(gt_classes, [gt_classes], summarize=100, message='gt_classes')
    with tf.variable_scope("FormatBoxGT"):
      return tf.one_hot((gt_classes + 1), depth=self.params.num_things_classes+1)


if __name__ == "__main__":
  labels = [True, True, True, False, False, False]
  indicator = [False, False, True, True, True, True]
  minibatch_size = 2
  positive_rate = 0.5

  rpn_sampler = RPNSampler(positive_rate=positive_rate)
  samples = rpn_sampler.sample(indicator, minibatch_size, labels)

  with tf.Session() as sess:
    samples_out = sess.run(samples)
    print(samples_out)