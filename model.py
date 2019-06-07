import os
import tensorflow as tf
from components import metrics
from components.feature_extractor import feature_extractor
from components.semantic_segmentation import SemanticSegmentationModel
from components.instance_segmentation import InstanceSegmentationModel
from components.panoptic_ops import merge_to_panoptic
from utils import summaries

class PanopticSegmentationModel(object):
  def __init__(self, images, labels, params, instance_gt=None):
    self.params = params
    self.images = images
    self.labels = labels
    self.instance_gt = instance_gt

    self.semantic_segmentation = SemanticSegmentationModel(self.params)
    self.instance_segmentation = InstanceSegmentationModel(self.params, is_training=params.is_training)

    self.apply_semantic_branch = params.apply_semantic_branch
    self.apply_instance_branch = params.apply_instance_branch

    self.groundtruth_dict = None
    if self.params.is_training:
      self.prepare_gt()

  def predict(self):
    features, _ = feature_extractor(None, self.images, self.params)

    prediction_dict = dict()

    if self.apply_semantic_branch:
      logits = self.semantic_segmentation.predict(features)
      prediction_dict['logits'] = logits
    if self.apply_instance_branch:
      prediction_dict.update(self.instance_segmentation.predict(features, prediction_dict, self.groundtruth_dict))

    return prediction_dict

  def postprocess(self, prediction_dict):

    if self.apply_semantic_branch:
      probs, predictions = self.semantic_segmentation.postprocess(prediction_dict['logits'])
      prediction_dict['semantic'] = predictions
      prediction_dict['semantic_probs'] = probs
    if self.apply_instance_branch:
      prediction_dict = self.instance_segmentation.postprocess(prediction_dict)

    if self.apply_semantic_branch and self.apply_instance_branch:
      prediction_dict = merge_to_panoptic(prediction_dict, self.params)

    return prediction_dict

  def loss(self, prediction_dict, save_image_summaries=True, save_eval_summaries=True):
    with tf.variable_scope("GetRegularization"):
      l2_regularizer = tf.contrib.layers.l2_regularizer(self.params.regularization_weight)

      reg_vars = [v for v in tf.trainable_variables() if 'weights' in v.name]
      reg_loss = tf.contrib.layers.apply_regularization(l2_regularizer, reg_vars)

    tf.summary.scalar('regularization', reg_loss, family='losses')

    losses_dict = dict()

    if self.apply_semantic_branch:
      logits = prediction_dict['logits']
      labels, weights = self.semantic_segmentation.format_gt(self.labels)
      sem_loss = self.semantic_segmentation.loss(logits=logits, labels=labels, weights=weights)
      losses_dict['semantic'] = sem_loss

      if save_image_summaries:
        summaries.image_summaries(self.images, labels, logits, weights, self.params)

      if save_eval_summaries:
        with tf.variable_scope('miou_training'):
          predictions = tf.nn.softmax(logits)
          predictions = tf.argmax(predictions, axis=-1)
          labels = tf.reshape(labels, [-1])
          predictions = tf.reshape(predictions, [-1])
          weights = tf.reshape(weights, [-1])
          total_cm = tf.confusion_matrix(labels=labels,
                                         predictions=predictions,
                                         num_classes=self.params.num_classes,
                                         weights=weights)
          miou = metrics.compute_mean_iou(total_cm)

        tf.summary.scalar('mIoU', tf.cast(miou, tf.float32), family='metrics')

    if self.apply_instance_branch:
      loss_dict = self.instance_segmentation.loss(prediction_dict, self.groundtruth_dict)
      losses_dict.update(loss_dict)

      if save_image_summaries:
        prediction_dict = self.instance_segmentation.postprocess(prediction_dict)
        self.gt_boxes = self.instance_gt['boxes']
        self.gt_num_boxes = self.instance_gt['num_boxes']
        summaries.image_summaries_boxes(self.images,
                                        self.gt_num_boxes,
                                        self.gt_boxes,
                                        prediction_dict,
                                        self.params)

    with tf.variable_scope("TotalLoss"):

      # Get total loss
      total_loss = reg_loss
      for loss_value in losses_dict.values():
        total_loss += loss_value

    tf.summary.scalar('total', total_loss, family='losses')

    return total_loss

  def prepare_gt(self):
    groundtruth_dict = dict()
    if self.apply_instance_branch:
      groundtruth_dict_ins = self.instance_segmentation.format_gt_dict(self.instance_gt)
      groundtruth_dict.update(groundtruth_dict_ins)
    if self.apply_semantic_branch:
      groundtruth_dict_sem = {'labels': self.labels}
      groundtruth_dict.update(groundtruth_dict_sem)
    self.groundtruth_dict = groundtruth_dict

  @staticmethod
  def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
      os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

  @staticmethod
  def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

if __name__ == '__main__':
  pass