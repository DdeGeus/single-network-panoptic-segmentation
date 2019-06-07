import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import box_utils

def batch_roi_pooling(features_batch, rois_normalized_batch, params):
  """

  Args:
    features_batch:
    rois_batch:
    img_shape:

  Returns:

  """
  with tf.variable_scope("ROIPooling"):
    rois_normalized_batch = tf.stop_gradient(rois_normalized_batch)

    rois_shape = tf.shape(rois_normalized_batch)
    ones_mat = tf.ones(rois_shape[:2], dtype=tf.int32)
    multiplier = tf.expand_dims(tf.range(start=0, limit=rois_shape[0]), 1)
    box_ind_batch = tf.reshape(ones_mat * multiplier, [-1])

    rois_normalized_batch = tf.map_fn(box_utils.convert_xyxy_to_yxyx_format, rois_normalized_batch)
    roi_features_cropped = tf.image.crop_and_resize(features_batch,
                                                    tf.reshape(rois_normalized_batch, [-1, 4]),
                                                    box_ind = box_ind_batch,
                                                    crop_size = params.roi_crop_size
                                                    )

    roi_features = slim.max_pool2d(roi_features_cropped,
                                   [params.roi_pool_kernel_size, params.roi_pool_kernel_size],
                                   stride=params.roi_pool_kernel_size)

  return roi_features
