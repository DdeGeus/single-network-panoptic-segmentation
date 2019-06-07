import tensorflow as tf

class BoxWrapper(object):
  def __init__(self, boxes):
    """
    Stores bounding boxes to enable batching with diff nums of boxes

    Args:
      boxes: Bounding boxes [N, 4], with order (xmin, ymin, xmax, ymax)
    """
    self.data = {'boxes': tf.cast(boxes, tf.float32)}

  def set_classes(self, classes):
    # TODO: should only be possible if shape is same is boxes
    self.data['classes'] = tf.cast(classes, tf.int32)

  def set_key(self, key, value, data_type=tf.int32):
    self.data[key] = tf.cast(value, data_type)

  def set_weights(self, weights):
    # TODO: should only be possible if shape is same is boxes
    self.data['weights'] = tf.cast(weights, tf.int32)

  def set_instance_ids(self, instance_ids):
    self.data['instance_ids'] = tf.cast(instance_ids, tf.int32)

  def replace_boxes(self, boxes):
    # TODO: should only be possible if shape is same as original and classes
    self.data['boxes'] = tf.cast(boxes, tf.float32)

  def get_boxes(self):
    """

    Returns: bounding boxes stored in wrapper

    """
    return self.data['boxes']

  def get_classes(self):
    return self.data['classes']

  def get_weights(self):
    return self.data['weights']

  def get_instance_ids(self):
    return self.data['instance_ids']

  def get_entry(self, key):
    return self.data[key]

  def get_all_data(self):
    return self.data

  def num_boxes(self):
    """

    Returns:

    """

    return tf.shape(self.data['boxes'])[0]

def init_box_wrappers_from_dataset(boxes, classes, weights, num_boxes, instance_ids):
  """
  Returns BoxWrapper based on data from dataset iterator

  Args:
    boxes:
    classes:
    weights:
    num_boxes:

  Returns:

  """
  with tf.variable_scope("InitBoxWrappers"):
    num_boxes = tf.reshape(num_boxes, [])
    box_wrapper = BoxWrapper(boxes[0:num_boxes])
    box_wrapper.set_classes(classes[0:num_boxes])
    box_wrapper.set_weights(weights[0:num_boxes])
    box_wrapper.set_instance_ids(instance_ids[0:num_boxes])
    # box_wrapper.set_key('instance_ids', instance_ids[0:num_boxes])

  return box_wrapper

