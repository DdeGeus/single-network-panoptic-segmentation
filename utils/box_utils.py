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

EPSILON=1e-7

"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

def get_center_coordinates_and_sizes(boxes):
  with tf.variable_scope("GetCenterCoordinatesAndSizes"):
    xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(boxes))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]

def encode_boxes(boxes, anchors, scale_factors=None):
  """Encode a box collection with respect to anchor collection.

  Args:
    boxes: BoxList holding N boxes to be encoded.
    anchors: BoxList of anchors.
    scale_factors: Factors to scale the encoded boxes (float, float, float, float).

  Returns:
    a tensor representing N anchor-encoded boxes of the format
    [ty, tx, th, tw].
  """
  with tf.variable_scope("EncodeBoxes"):
    # Convert anchors and boxes to the center coordinate representation.
    xmin_a, ymin_a, xmax_a, ymax_a = tf.unstack(tf.transpose(anchors))
    wa = xmax_a - xmin_a
    ha = ymax_a - ymin_a
    ycenter_a = ymin_a + ha / 2.
    xcenter_a = xmin_a + wa / 2.

    xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(boxes))
    w = xmax - xmin
    h = ymax - ymin
    ycenter = ymin + h / 2.
    xcenter = xmin + w / 2.

    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)

    # Scales location targets as used in paper for joint training.
    if scale_factors:
      ty *= scale_factors[0]
      tx *= scale_factors[1]
      th *= scale_factors[2]
      tw *= scale_factors[3]

    return tf.transpose(tf.stack([ty, tx, th, tw]))

def decode_boxes(encoded_boxes, anchors, scale_factors=None):
  """Decode relative codes to boxes.

  Args:
    encoded_boxes: encoded boxes with relative coding to anchors [N, 4]
    anchors: anchors [N, 4]
    scale_factors: Factors to scale the decoded boxes (float, float, float, float).

  Returns:
    boxes: decoded boxes [N, 4]
  """

  with tf.variable_scope("DecodeBoxes"):
    xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(anchors))
    wa = xmax - xmin
    ha = ymax - ymin
    ycenter_a = ymin + ha / 2.
    xcenter_a = xmin + wa / 2.

    ty, tx, th, tw = tf.unstack(tf.transpose(encoded_boxes))

    if scale_factors:
      ty /= scale_factors[0]
      tx /= scale_factors[1]
      th /= scale_factors[2]
      tw /= scale_factors[3]

    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))

def normalize_boxes(boxes, orig_height, orig_width):
  """

  Args:
    boxes: input boxes [N, 4] [x_min, y_min, x_max, y_max]
    orig_height: original image height for input boxes
    orig_width: original image width for input boxes

  Returns: normalized boxes

  """
  with tf.variable_scope("NormalizeBoxes"):
    orig_height = tf.cast(orig_height, tf.float32)
    orig_width = tf.cast(orig_width, tf.float32)
    boxes = tf.cast(boxes, tf.float32)
    x_min, y_min, x_max, y_max = tf.split(boxes, num_or_size_splits=4, axis=1)
    x_min = x_min / orig_width
    y_min = y_min / orig_height
    x_max = x_max / orig_width
    y_max = y_max / orig_height

    return tf.concat([x_min, y_min, x_max, y_max], axis=1)

def resize_normalized_boxes(norm_boxes, new_height, new_width):
  """
  Resize normalized boxes to a given set of coordinates

  Args:
    norm_boxes: normalized boxes [N, 4] [x_min, y_min, x_max, y_max] (between 0 and 1)
    new_height: new height for the normalized boxes
    new_width: new width for the normalized boxes

  Returns: Resized boxes

  """
  with tf.variable_scope("ResizeNormBoxes"):
    x_min, y_min, x_max, y_max = tf.split(norm_boxes, num_or_size_splits=4, axis=1)
    x_min = x_min * new_width
    y_min = y_min * new_height
    x_max = x_max * new_width
    y_max = y_max * new_height

    return tf.concat([x_min, y_min, x_max, y_max], axis=1)

def flip_normalized_boxes_left_right(boxes):
  """
  Flips boxes that are already normalized from left to right

  Args:
    boxes: normalized boxes

  Returns: Flipped boxes

  """
  with tf.variable_scope("FlipBoxesLeftRight"):
    boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1],
                      1 - boxes[:, 0], boxes[:, 3]], axis=-1)

    return boxes

def convert_input_box_format(boxes):
  with tf.variable_scope("ConvertInputBoxFormat"):
    boxes = tf.reshape(boxes, [-1, 4])
    return tf.transpose([boxes[:, 0],
                         boxes[:, 1],
                         boxes[:, 0]+boxes[:, 2],
                         boxes[:, 1]+boxes[:, 3]])

def calculate_ious(boxes_1, boxes_2):

  with tf.variable_scope("CalculateIous"):
    x_min_1, y_min_1, x_max_1, y_max_1 = tf.split(boxes_1, 4, axis=1)
    x_min_2, y_min_2, x_max_2, y_max_2 = tf.unstack(boxes_2, axis=1)

    max_x_min = tf.maximum(x_min_1, x_min_2)
    max_y_min = tf.maximum(y_min_1, y_min_2)

    min_x_max = tf.minimum(x_max_1, x_max_2)
    min_y_max = tf.minimum(y_max_1, y_max_2)

    x_overlap = tf.maximum(0., min_x_max - max_x_min)
    y_overlap = tf.maximum(0., min_y_max - max_y_min)

    overlaps = x_overlap * y_overlap

    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    ious = overlaps / (area_1 + area_2 - overlaps)

    return ious

def calculate_ious_2(boxes_1, boxes_2):
  with tf.variable_scope("CalculateIous"):
    x_min_1, y_min_1, x_max_1, y_max_1 = tf.split(boxes_1, 4, axis=1)
    x_min_2, y_min_2, x_max_2, y_max_2 = tf.split(boxes_2, 4, axis=1)
    x_min_2 = tf.squeeze(x_min_2, 1)
    y_min_2 = tf.squeeze(y_min_2, 1)
    x_max_2 = tf.squeeze(x_max_2, 1)
    y_max_2 = tf.squeeze(y_max_2, 1)

    max_x_min = tf.maximum(x_min_1, x_min_2)
    max_y_min = tf.maximum(y_min_1, y_min_2)

    min_x_max = tf.minimum(x_max_1, x_max_2)
    min_y_max = tf.minimum(y_max_1, y_max_2)

    x_overlap = tf.maximum(0., min_x_max - max_x_min)
    y_overlap = tf.maximum(0., min_y_max - max_y_min)

    overlaps = x_overlap * y_overlap

    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    ious = overlaps / (area_1 + area_2 - overlaps)

    return ious

def clip_to_img_boundaries(boxes, image_shape):
  """

  Args:
    boxes: decoded boxes with relative coding to anchors [N, 4]
    image_shape: shape of the image [2], (height, width)

  Returns:
    Boxes that have been clipped to the image boundaries [N, 4]

  """
  with tf.variable_scope("ClipToImgBoundaries"):
    xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(boxes))
    hi, wi = tf.cast(image_shape[0], tf.float32), tf.cast(image_shape[1], tf.float32)

    # xmin = tf.maximum(tf.minimum(xmin, wi - 1.), 0.)
    # ymin = tf.maximum(tf.minimum(ymin, hi - 1.), 0.)
    #
    # xmax = tf.maximum(tf.minimum(xmax, wi - 1.), 0.)
    # ymax = tf.maximum(tf.minimum(ymax, hi - 1.), 0.)
    xmin = tf.maximum(tf.minimum(xmin, wi), 0.)
    ymin = tf.maximum(tf.minimum(ymin, hi), 0.)

    xmax = tf.maximum(tf.minimum(xmax, wi), 0.)
    ymax = tf.maximum(tf.minimum(ymax, hi), 0.)

    return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))

def convert_xyxy_to_yxyx_format(boxes):
  with tf.variable_scope("ConvertXyxyToYxyxFormat"):
    xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(boxes))
    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))

def convert_yxyx_to_xyxy_format(boxes):
  with tf.variable_scope("ConvertYxyxToXyxyFormat"):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(boxes))
    return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))

def pad_boxes_and_return_num(boxes, pad_size):
  with tf.variable_scope("PadBoxesReturnNum"):
    num_boxes = tf.shape(boxes)[0]
    shape = [[0, pad_size - num_boxes], [0, 0]]
    boxes_pad = tf.pad(boxes, shape)

    return boxes_pad, num_boxes