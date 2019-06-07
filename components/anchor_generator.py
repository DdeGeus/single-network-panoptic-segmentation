import tensorflow as tf
import numpy as np
from utils import box_utils

def generate(base_size,
             stride,
             scales,
             ratios,
             features_height,
             features_width,
             offset=None):
  """

  Args:
    base_size: (height, width)
    stride: (height, width)
    scales: (height, width)
    ratios: (height, width)
    features_height:
    features_width:
    offset: (height, width)

  Returns:

  """

  with tf.variable_scope('anchor_generator'):
    if offset is None:
      offset = [stride[0]/2, stride[1]/2]

    features_width = tf.cast(features_width, tf.int32)
    features_height = tf.cast(features_height, tf.int32)
    scales = tf.convert_to_tensor(scales, dtype=tf.float32)
    ratios = tf.convert_to_tensor(ratios, dtype=tf.float32)
    offset = tf.convert_to_tensor(offset, dtype=tf.float32)

    scales_grid, ratios_grid = tf.meshgrid(scales,
                                           ratios)

    scales_grid = tf.reshape(scales_grid, [-1, 1])
    ratios_grid = tf.reshape(ratios_grid, [-1, 1])

    ratio_sqrts = tf.sqrt(ratios_grid)

    heights = scales_grid / ratio_sqrts * base_size[1]
    widths = scales_grid * ratio_sqrts * base_size[0]

    x_centers = tf.cast(tf.range(features_width), tf.float32)
    x_centers = x_centers * stride[1]
    y_centers = tf.cast(tf.range(features_height), tf.float32)
    y_centers = y_centers * stride[0]
    # x_centers = x_centers + offset[1]
    # y_centers = y_centers + offset[0]

    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    widths, x_centers = tf.meshgrid(widths, x_centers)
    heights, y_centers = tf.meshgrid(heights, y_centers)

    anchor_centers = tf.stack([x_centers, y_centers], axis=2)
    anchor_centers = tf.reshape(anchor_centers, [-1, 2])

    anchor_sizes = tf.stack([widths, heights], axis=2)
    anchor_sizes = tf.reshape(anchor_sizes, [-1, 2])

    anchors = tf.concat([anchor_centers - .5 * anchor_sizes,
                         anchor_centers + .5 * anchor_sizes], 1)

    # anchors = box_utils.convert_yxyx_to_xyxy_format(anchors)

    return anchors


if __name__ == '__main__':
  anchor_size = [128, 128]
  anchor_stride = [8, 8]
  anchor_offset = [0, 0]
  anchor_scales = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
  anchor_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
  height, width = 64, 128
  anchors = generate(anchor_size,
                     anchor_stride,
                     anchor_scales,
                     anchor_ratios,
                     height,
                     width)

  init = tf.global_variables_initializer()
  with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(init)
    anchors_out = sess.run(anchors)
    print(anchors_out[-30:])
    print(anchors.shape)
    print(anchors_out[158623])

