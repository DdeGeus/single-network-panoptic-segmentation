import tensorflow as tf

VOID_CLASS = 255
VARIABLE_STUFF_HEUR_VALUE = 2048
HEURISTIC_PROB_TH = 0.25

def merge_to_panoptic(prediction_dict, params):
  probs_batch = tf.unstack(prediction_dict['semantic_probs'], num=params.Nb)
  instance_masks_batch = tf.unstack(prediction_dict['masks_reshaped_probs'], num=params.Nb)
  instance_classes_batch = tf.unstack(prediction_dict['det_class_postprocessed'], num=params.Nb)
  num_boxes_batch = tf.unstack(prediction_dict['det_num_boxes'], num=params.Nb)

  pan_preds = list()

  for probs, instance_masks, instance_classes, num_boxes in zip(
          probs_batch, instance_masks_batch, instance_classes_batch, num_boxes_batch):

    # REMOVE OVERLAP AND THRESHOLD
    instance_masks = instance_masks[:num_boxes]

    def get_instance_masks(instance_masks):
      mask_max = tf.reduce_max(instance_masks, axis=0)
      total_probs = instance_masks
      mask_argmax = tf.argmax(total_probs, axis=0)
      mask_mask = tf.greater_equal(mask_max, 0.5)

      indices = mask_argmax
      depth = tf.shape(instance_masks)[0]
      masks_out = tf.one_hot(indices, depth, axis=0)
      instance_masks = tf.cast(masks_out, tf.uint8) * tf.cast(mask_mask, tf.uint8)

      return instance_masks

    cond = tf.less(num_boxes, 1)
    instance_masks = tf.cond(cond,
                             true_fn=lambda: tf.zeros((1, params.height_input, params.width_input), dtype=tf.uint8),
                             false_fn=lambda: get_instance_masks(instance_masks))

    instance_masks = instance_masks[:num_boxes]
    instance_classes = instance_classes[:num_boxes]

    probs = tf.image.resize_images(probs, tf.shape(instance_masks)[1:3],
                                   method=tf.image.ResizeMethod.BILINEAR)

    # Retrieve background semantic segmentation maps
    multiplier = tf.cast(tf.logical_not(tf.cast(params.cids2is_object, tf.bool)), tf.float32)
    multiplier = tf.reshape(multiplier, [1, 1, -1])

    probs_new = tf.multiply(probs, multiplier)
    probs_max = tf.reduce_max(probs_new, 2)
    probs_above_th_bool = tf.greater(probs_max, HEURISTIC_PROB_TH)

    seg_mask = tf.expand_dims(tf.cast(tf.argmax(probs_new, 2), tf.int32), -1)
    seg_mask = tf.squeeze(seg_mask)

    probs_above_th = tf.cast(probs_above_th_bool, tf.int32)
    probs_below_th = tf.cast(tf.logical_not(probs_above_th_bool), tf.int32)
    void_mask_below_th = tf.multiply(probs_below_th, VOID_CLASS)
    seg_mask = tf.add(tf.multiply(probs_above_th, seg_mask), void_mask_below_th)

    seg_mask_tmp = tf.reshape(seg_mask, [-1])
    seg_mask_tmp = tf.one_hot(seg_mask_tmp, depth=len(params.cids2is_object))

    img_shape = tf.shape(instance_masks)[1:3]
    image_size = img_shape[0] * img_shape[1]
    th_count = image_size / VARIABLE_STUFF_HEUR_VALUE
    th_count = tf.cast(th_count, tf.float32)

    seg_count = tf.reduce_sum(seg_mask_tmp, axis=0)
    multiplier_2 = tf.cast(tf.greater(seg_count, th_count), tf.float32)

    multiplier_total = tf.multiply(multiplier, multiplier_2)

    probs_new = tf.multiply(probs, multiplier_total)
    probs_max = tf.reduce_max(probs_new, 2)
    probs_above_th_bool = tf.greater(probs_max, HEURISTIC_PROB_TH)

    seg_mask = tf.expand_dims(tf.cast(tf.argmax(probs_new, 2), tf.int32), -1)
    seg_mask = tf.squeeze(seg_mask)

    probs_above_th = tf.cast(probs_above_th_bool, tf.int32)
    probs_below_th = tf.cast(tf.logical_not(probs_above_th_bool), tf.int32)
    void_mask_below_th = tf.multiply(probs_below_th, VOID_CLASS)
    seg_mask = tf.add(tf.multiply(probs_above_th, seg_mask), void_mask_below_th)

    # Store ids for non-object classes
    ids = tf.ones_like(seg_mask)

    # ids = tf.Print(ids, [tf.shape(ids)], message='ids', summarize=100)

    # Retrieve the semantic segmentation predictions for the object classes only
    decs_seg = tf.expand_dims(tf.cast(tf.argmax(probs, 2), tf.int32), -1)
    decs_gather_params = tf.multiply(tf.range(len(params.cids2is_object)), params.cids2is_object)
    decs_obj = tf.squeeze(tf.gather(decs_gather_params, decs_seg), axis=2)

    # Get sum of instance masks in binary format
    masks_ins_total = tf.squeeze(tf.cast(tf.reduce_sum(instance_masks, axis=0), tf.int32))
    split_mask = tf.squeeze(tf.contrib.image.connected_components(decs_obj))

    # Retrieve and gather masks of objects not detected by instance segmentation
    mask_nums, _ = tf.unique(tf.reshape(masks_ins_total * split_mask, [-1]))

    # Retrieve the instance masks for object classes
    objectcids2cids = tf.gather(params.labels2cids, params.object_cids2lids)
    instance_classes = tf.gather(objectcids2cids, instance_classes)
    # instance_classes = tf.Print(instance_classes, [instance_classes], message='instance_classes',
    #                             summarize=100)
    instance_classes = tf.reshape(instance_classes, [-1, 1, 1])
    # instance_masks = tf.Print(instance_masks, [tf.reduce_max(instance_masks)], message='max_instance_masks')
    instance_masks_classes = tf.multiply(tf.cast(instance_masks, tf.int32), instance_classes)
    # instance_masks_classes = tf.Print(instance_masks_classes, [tf.reduce_max(instance_masks_classes)], message='max_instance_masks_classes')
    total_mask_bool = tf.reduce_sum(instance_masks, axis=0)
    # total_mask_bool = tf.Print(total_mask_bool, [tf.reduce_max(total_mask_bool)], message='total_mask_bool')
    total_mask = tf.reduce_sum(instance_masks_classes, axis=0)
    # total_mask = tf.Print(total_mask, [tf.reduce_max(total_mask)], message='max_total_mask')
    total_mask_where = tf.cast(tf.logical_not(tf.greater(total_mask_bool, 0)), tf.int32)

    # total_mask_where = tf.Print(total_mask_where, [tf.shape(total_mask_where)], message='total_mask_where', summarize=100)
    # total_mask_where = tf.Print(total_mask_where, [tf.reduce_max(total_mask_where)],
    #                                   message='max_total_mask_where')

    # Retrieve and store ids for detected object instances
    ids_instance_nums = tf.range(num_boxes) + (tf.reduce_max(ids) + 1)
    ids_instances = tf.multiply(tf.cast(instance_masks, tf.int32), tf.reshape(ids_instance_nums, [-1, 1, 1]))

    # total_mask_where = tf.Print(total_mask_where, [tf.shape(total_mask_where)], message='total_mask_where',
    #                             summarize=100)

    ids_instances = tf.reduce_sum(ids_instances, axis=0)
    ids = tf.add(tf.multiply(ids, total_mask_where), ids_instances)

    seg_mask = tf.add(tf.multiply(seg_mask, total_mask_where), total_mask)
    # seg_mask = tf.Print(seg_mask, [tf.reduce_max(seg_mask)],
    #                                   message='seg_mask')

    pan = tf.stack([seg_mask, ids], axis=2)

    pan_preds.append(pan)

  pan_preds = tf.stack(pan_preds)
  pan_preds = tf.image.resize_nearest_neighbor(pan_preds, [params.height_orig, params.width_orig])

  prediction_dict['panoptic'] = pan_preds

  return prediction_dict