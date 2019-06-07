import tensorflow as tf
import numpy as np
from functools import partial

from utils import box_utils

def retrieve_rpn_additions(seg_logits, num_additions, params):
  with tf.control_dependencies([seg_logits]):
    seg_logits_batch = tf.unstack(seg_logits, num=params.Nb)
    proposal_boxes_batch = list()
    box_scores_batch = list()
    valid_batch = list()

    dummy_min = tf.cast([[0, 0]], tf.int64)
    dummy_max = tf.cast([[1, 1]], tf.int64)

    for seg_logits in seg_logits_batch:
      def _get_corner_pixels(one_hot_tensor):
        xy = tf.where(one_hot_tensor)

        y = xy[..., 0]
        x = xy[..., 1]

        x_min = tf.reduce_min(x)
        # x_max = tf.reduce_max(x)
        x_max = tf.reduce_max(x) + 1
        y_min = tf.reduce_min(y)
        # y_max = tf.reduce_max(y)
        y_max = tf.reduce_max(y) + 1

        # return [x_min, y_min, x_max, y_max]
        return [y_min, x_min, y_max, x_max]

      probs = tf.nn.softmax(seg_logits)
      img_tensor = tf.cast(tf.argmax(probs, 2), tf.int32)

      img_objects = tf.gather(params.cids2is_object, tf.cast(img_tensor, tf.int32))
      img_only_objects = tf.multiply(tf.cast(img_tensor, tf.int32), img_objects)
      connected_components = tf.contrib.image.connected_components(img_only_objects)
      init_num_components = tf.reduce_max(connected_components) + 1

      max_components = tf.minimum(init_num_components, num_additions - 2)
      bool_mask = tf.less_equal(connected_components, max_components)
      connected_components = tf.multiply(tf.cast(bool_mask, tf.int32), connected_components)

      num_components = tf.reduce_max(connected_components) + 1
      connected_one_hot = tf.one_hot(connected_components, depth=num_components)

      one_hot_reshaped = tf.transpose(connected_one_hot, [2, 0, 1])

      bboxes = tf.map_fn(_get_corner_pixels, one_hot_reshaped, dtype=[tf.int64, tf.int64, tf.int64, tf.int64])
      bboxes = (tf.transpose(bboxes, [1, 0]))

      num_boxes = num_components
      num_paddings = num_additions - num_boxes

      minima = bboxes[..., 0:2]
      maxima = bboxes[..., 2:4]

      minima = tf.concat([minima, dummy_min], axis=0)
      maxima = tf.concat([maxima, dummy_max], axis=0)

      maxval = tf.maximum(num_boxes, 1)

      min_rand = tf.random_uniform([num_paddings], minval=0, maxval=maxval, dtype=tf.int32, seed=1)
      max_rand = tf.random_uniform([num_paddings], minval=0, maxval=maxval, dtype=tf.int32, seed=1)

      new_minima = tf.gather(minima, min_rand)
      new_maxima = tf.gather(maxima, max_rand)

      extra_boxes = tf.concat([new_minima, new_maxima], axis=1)

      total_boxes = tf.concat([bboxes, extra_boxes], axis=0)
      proposal_boxes_batch.append(total_boxes)

      bboxes_scores = tf.ones((num_boxes, 1), dtype=tf.float32) * 0.9
      extra_boxes_scores = tf.ones((num_paddings, 1), dtype=tf.float32) * 0.5

      box_scores = tf.concat([bboxes_scores, extra_boxes_scores], axis=0)
      box_scores_batch.append(box_scores)

    proposal_boxes = tf.stack(proposal_boxes_batch)

    return tf.cast(proposal_boxes, tf.float32)


def expand_instance_bboxes(inst_bboxes_batch, inst_classes_batch, seg_logits, params, max_ratio=0.25, iou_th=0.5):
  with tf.control_dependencies([seg_logits]):

    seg_logits_batch = tf.unstack(seg_logits, num=params.Nb)
    inst_bboxes_batch = tf.unstack(inst_bboxes_batch, num=params.Nb)
    inst_classes_batch = tf.unstack(inst_classes_batch, num=params.Nb)

    ext_bbox_lists = list()
    objectcids2cids = tf.gather(params.labels2cids, params.object_cids2lids)

    for seg_logits, inst_bboxes, inst_classes in zip(seg_logits_batch, inst_bboxes_batch, inst_classes_batch):
      probs = tf.nn.softmax(seg_logits)
      img_tensor = tf.cast(tf.argmax(probs, 2), tf.int32)
      inst_seg_classes = tf.gather(objectcids2cids, inst_classes)

      img_objects = tf.gather(params.cids2is_object, tf.cast(img_tensor, tf.int32))
      img_only_objects = tf.multiply(tf.cast(img_tensor, tf.int32), img_objects)
      connected_components = tf.contrib.image.connected_components(img_only_objects)

      def _get_corner_pixels(one_hot_tensor):
        xy = tf.where(one_hot_tensor)

        y = xy[..., 0]
        x = xy[..., 1]

        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)
        y_min = tf.reduce_min(y)
        y_max = tf.reduce_max(y)

        return [y_min, x_min, y_max, x_max]

      seg_class_and_id = connected_components * 1000 + img_only_objects
      unique_seg_cl_id, _ = tf.unique(tf.reshape(seg_class_and_id, [-1]))
      seg_classes = tf.mod(unique_seg_cl_id, 1000)

      connected_one_hot = tf.one_hot(connected_components, depth=tf.reduce_max(connected_components)+1)

      one_hot_reshaped = tf.transpose(connected_one_hot, [2, 0, 1])

      seg_bboxes = tf.map_fn(_get_corner_pixels, one_hot_reshaped, dtype=[tf.int64, tf.int64, tf.int64, tf.int64],
                             parallel_iterations=16)
      seg_bboxes = tf.transpose(seg_bboxes, [1, 0])

      def expand_boxes_np(ious, det_boxes, det_classes, seg_boxes, seg_classes):
        det_new_boxes = list()
        seg_boxes = seg_boxes[1:]
        seg_classes = seg_classes[1:]
        ious = ious[..., 1:]

        for i, (det_class, det_box) in enumerate(zip(det_classes, det_boxes)):
          valid_ious = ious[i][seg_classes == det_class]
          valid_seg_boxes = seg_boxes[seg_classes == det_class]
          det_new_box = det_box
          if len(valid_ious.tolist()) > 0:
            if np.max(valid_ious) > iou_th:
              argmax = np.argmax(valid_ious)
              ref_seg_box = valid_seg_boxes[argmax]

              xmin_ref = ref_seg_box[0]
              ymin_ref = ref_seg_box[1]
              xmax_ref = ref_seg_box[2]
              ymax_ref = ref_seg_box[3]

              xmin_orig = det_box[0]
              ymin_orig = det_box[1]
              xmax_orig = det_box[2]
              ymax_orig = det_box[3]

              orig_width = xmax_orig - xmin_orig
              orig_height = ymax_orig - ymin_orig
              width_limit = orig_width * max_ratio
              height_limit = orig_height * max_ratio

              xmin_limit = xmin_orig - width_limit
              ymin_limit = ymin_orig - height_limit
              xmax_limit = xmax_orig + width_limit
              ymax_limit = ymax_orig + height_limit

              x_min = np.minimum(np.maximum(xmin_limit, xmin_ref), xmin_orig)
              y_min = np.minimum(np.maximum(ymin_limit, ymin_ref), ymin_orig)
              x_max = np.maximum(np.minimum(xmax_limit, xmax_ref), xmax_orig)
              y_max = np.maximum(np.minimum(ymax_limit, ymax_ref), ymax_orig)

              det_new_box = np.array([x_min.astype(np.float32), y_min.astype(np.float32), x_max.astype(np.float32),
                                      y_max.astype(np.float32)])
          det_new_boxes.append(det_new_box)

        det_boxes = np.array(det_new_boxes)

        return det_boxes

      ious = box_utils.calculate_ious_2(inst_bboxes, tf.cast(seg_bboxes, tf.float32))

      ext_bboxes = tf.py_func(expand_boxes_np, [ious, inst_bboxes, inst_seg_classes, seg_bboxes, seg_classes], Tout=tf.float32)
      ext_bboxes.set_shape([None, 4])

      ext_bbox_lists.append(ext_bboxes)

    ext_bboxes_batch = tf.stack(ext_bbox_lists)
    return ext_bboxes_batch