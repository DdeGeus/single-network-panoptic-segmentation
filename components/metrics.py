import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from utils import box_utils

class RecallEvaluator(object):

  def __init__(self, iou_th=0.5):
    self.tp_count = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    self.fn_count = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    self.iou_th = iou_th

  def update(self, det_boxes, gt_boxes):
    """

    Args:
      det_boxes: Normalized detected boxes [N, 4]
      gt_boxes: Normalized detected boxes [M, 4]

    Returns:
      Update operations for updating true positives and falsenegatives

    """
    ious = box_utils.calculate_ious(det_boxes, gt_boxes)
    max_ious = tf.reduce_max(ious, axis=1)
    matches = tf.argmax(ious, axis=1)
    tp_indicator = tf.greater_equal(max_ious, self.iou_th)
    tp_matches = tf.boolean_mask(matches, tp_indicator)
    tp_unique, _ = tf.unique(tp_matches)
    tp = tf.shape(tp_unique)[0]
    fn = tf.shape(gt_boxes)[0] - tp
    update_op_tp = self.tp_count.assign_add(tp)
    update_op_fn = self.fn_count.assign_add(fn)
    return [update_op_tp, update_op_fn]

  def calculate_recall(self):
    recall = self.tp_count / (self.tp_count+self.fn_count)
    return recall

def evaluate_mean_recall(det_boxes, gt_boxes, iou_th=0.5):
  """

  Args:
    det_boxes: Normalized detected boxes [N, 4]
    gt_boxes: Normalized detected boxes [M, 4]

  Returns:
    Update operations for updating true positives and falsenegatives

  """
  ious = box_utils.calculate_ious(det_boxes, gt_boxes)
  max_ious = tf.reduce_max(ious, axis=1)
  matches = tf.argmax(ious, axis=1)
  tp_indicator = tf.greater_equal(max_ious, iou_th)
  tp_matches = tf.boolean_mask(matches, tp_indicator)
  tp_unique, _ = tf.unique(tp_matches)
  tp = tf.shape(tp_unique)[0]
  fn = tf.shape(gt_boxes)[0] - tp

  mean_recall = tp / (tp + fn)

  return mean_recall

def compute_mean_iou(total_cm):
  """Compute the mean intersection-over-union via the confusion matrix."""
  sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
  sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
  cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
  denominator = sum_over_row + sum_over_col - cm_diag

  # The mean is only computed over classes that appear in the
  # label or prediction tensor. If the denominator is 0, we need to
  # ignore the class.
  num_valid_entries = math_ops.reduce_sum(
    math_ops.cast(
      math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

  # If the value of the denominator is 0, set it to 1 to avoid
  # zero division.
  denominator = array_ops.where(
    math_ops.greater(denominator, 0), denominator,
    array_ops.ones_like(denominator))
  iou = math_ops.div(cm_diag, denominator)

  # If the number of valid entries is 0 (no classes) we return 0.
  result = array_ops.where(
    math_ops.greater(num_valid_entries, 0),
    math_ops.reduce_sum(iou) / num_valid_entries, 0)
  return result