import tensorflow as tf
import functools
from utils import box_utils

def image_summaries(images, labels, logits, weights, params):
  with tf.variable_scope("ImageSummaries"):
    color_label = tf.gather(tf.cast(params.cids2colors, tf.uint8), labels)
    predictions = tf.nn.softmax(logits)
    predictions = tf.argmax(predictions, axis=-1)

    color_pred = tf.gather(tf.cast(params.cids2colors, tf.uint8), predictions)
  tf.summary.image('image', images)
  tf.summary.image('ground_truth', color_label * tf.expand_dims(tf.cast(weights, tf.uint8), -1))
  tf.summary.image('prediction', color_pred)

def image_summaries_boxes(images, gt_num_boxes, gt_boxes, prediction_dict, params):
  tf.summary.image('image', images)

  with tf.variable_scope("ImageSummariesBoxes"):

    def _box_draw_fn(inputs):
      image = tf.expand_dims(inputs[0], 0)
      num_boxes = tf.cast(tf.reshape(inputs[1], []), tf.int32)
      boxes = tf.expand_dims(inputs[2][0:num_boxes], 0)
      boxes_tf = box_utils.convert_xyxy_to_yxyx_format(boxes)
      image_with_gt = tf.image.draw_bounding_boxes(image, tf.cast(boxes_tf, tf.float32))
      image_with_gt = tf.squeeze(image_with_gt, 0)
      return image_with_gt

    images_with_gt = tf.map_fn(_box_draw_fn, [images, gt_num_boxes, gt_boxes],
                               dtype=tf.float32)

    boxes = prediction_dict['rpn_boxes_postprocessed']

    _normalize_fn = functools.partial(box_utils.normalize_boxes,
                                      orig_height=params.height_input,
                                      orig_width=params.width_input)
    boxes_normalized = tf.map_fn(_normalize_fn, boxes)

    num_boxes = tf.cast(params.Nb * [20], tf.int32)
    images_with_pred = tf.map_fn(_box_draw_fn, [images, num_boxes, boxes_normalized],
                                dtype=tf.float32)

    boxes_det = prediction_dict['det_boxes_postprocessed']
    _normalize_fn = functools.partial(box_utils.normalize_boxes,
                                      orig_height=params.height_input,
                                      orig_width=params.width_input)
    boxes_det_normalized = tf.map_fn(_normalize_fn, boxes_det)
    num_boxes = prediction_dict['det_num_boxes']
    # num_boxes = tf.cast(params.Nb * [10], tf.int32)
    images_with_det_pred = tf.map_fn(_box_draw_fn, [images, num_boxes, boxes_det_normalized],
                                dtype=tf.float32)

  tf.summary.image('ground_truth', images_with_gt)
  tf.summary.image('RPN_prediction', images_with_pred)
  tf.summary.image('det_prediction', images_with_det_pred)
