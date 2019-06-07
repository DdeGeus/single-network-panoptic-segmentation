import tensorflow as tf
import functools
from utils import box_utils

def extract_masks_by_id(masks, ids, boxes, params, crop_size=(33, 33)):
  """

  Args:
    masks: [Nb, H, W]
    ids: [Nb, num_boxes]
    boxes: [Nb, num_boxes, 4]
    crop_size: [2] (H, W)

  Returns:
    Extracted and cropped masks [Nb, crop_size[0], crop_size[1], num_boxes]

  """
  def _compare_and_crop(inputs, mask):
    with tf.variable_scope("CompareAndCropPerBbox"):
      identifier = inputs[0]
      box = inputs[1]
      mask_equal = tf.cast(tf.equal(mask, identifier), tf.float32)
      box = tf.reshape(box, [1, -1])
      box_formatted = box_utils.convert_xyxy_to_yxyx_format(box)
      mask_crop = tf.image.crop_and_resize(tf.reshape(mask_equal, [1, tf.shape(mask)[0], tf.shape(mask)[1], 1]),
                                           box_formatted,
                                           box_ind=[0],
                                           crop_size=crop_size)
      return tf.reshape(mask_crop, [crop_size[0], crop_size[1]])

  def _compare_and_crop_batch(inputs):
    with tf.variable_scope("CompareAndCropPerImage"):
      mask = inputs[2]
      _compare_and_crop_fn = functools.partial(_compare_and_crop, mask=mask)
      return tf.map_fn(_compare_and_crop_fn, inputs[0:2], dtype=tf.float32)

  with tf.variable_scope("ExtractBoxesById"):
    inputs = [ids, boxes, masks]

    return tf.map_fn(_compare_and_crop_batch, inputs, tf.float32)

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  def transform_boxes_relative_to_boxes(boxes, reference_boxes):
    boxes = tf.reshape(boxes, [-1, 2, 2])
    min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
    max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
    transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
    return tf.reshape(transformed_boxes, [-1, 4])

  box_masks = tf.expand_dims(box_masks, axis=3)
  num_boxes = tf.shape(box_masks)[0]
  unit_boxes = tf.concat(
      [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
  boxes = box_utils.convert_xyxy_to_yxyx_format(boxes)
  reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
  image_masks = tf.image.crop_and_resize(image=box_masks,
                                         boxes=reverse_boxes,
                                         box_ind=tf.range(num_boxes),
                                         crop_size=[image_height, image_width],
                                         extrapolation_value=0.0)
  return tf.squeeze(image_masks, axis=3)