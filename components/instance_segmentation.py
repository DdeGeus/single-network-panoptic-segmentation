import tensorflow as tf
import tensorflow.contrib.slim as slim
import functools
from components import anchor_generator, losses, box_matcher, sampler, roi_pooling
from components import feature_extractor, explicit_info_exchange
from utils import box_utils, box_wrapper, mask_utils

class InstanceSegmentationModel(object):
  def __init__(self, params, is_training):
    self.params = params
    self.is_training = is_training

    self._anchor_size = [params.base_anchor, params.base_anchor]
    self._anchor_stride = [8, 8]
    self._anchor_offset = [0, 0]
    self._anchor_scales = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
    self._anchor_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]

    self.num_anchors_per_location = len(self._anchor_scales) * len(self._anchor_ratios)

    self.rpn_minibatch_size = 256
    self.rpn_postive_rate = 0.5
    if params.rpn_roi_settings == 'old':
      self.rpn_iou_upper_th = 0.5
      self.rpn_iou_lower_th = 0.1
      self.rpn_nms_iou_th = 0.5
    elif params.rpn_roi_settings == 'new':
      self.rpn_iou_upper_th = 0.7
      self.rpn_iou_lower_th = 0.3
      self.rpn_nms_iou_th = 0.7
    elif params.rpn_roi_settings == 'hybrid':
      self.rpn_iou_upper_th = 0.6
      self.rpn_iou_lower_th = 0.2
      self.rpn_nms_iou_th = 0.6
    else:
      self.rpn_iou_upper_th = 0.5
      self.rpn_iou_lower_th = 0.1
      self.rpn_nms_iou_th = 0.5

    self.rpn_nms_num_samples = 1000

    self.rpn_box_matcher = box_matcher.BoxMatcher(self.rpn_iou_upper_th,
                                                  self.rpn_iou_lower_th)
    self.rpn_sampler = sampler.RPNSampler(positive_rate=self.rpn_postive_rate)

    self.rpn_regression_loss_weight = params.rpn_regression_loss_weight
    self.rpn_classification_loss_weight = params.rpn_classification_loss_weight

    self.PREDICT_DETECTION = True
    self.MASK_PREDICTION = True

    self.IMPLICIT_INFO_EXCHANGE = True
    self.ADD_REGION_PROPOSALS = True
    self.EXPAND_BOXES = True

    self.num_things_classes = 8

    self.roi_crop_size = [14, 14]
    self.roi_pool_kernel_size = 2

    if params.rpn_roi_settings == 'old':
      self.roi_minibatch_size = 64
    elif params.rpn_roi_settings == 'new':
      self.roi_minibatch_size = 256
    elif params.rpn_roi_settings == 'hybrid':
      self.roi_minibatch_size = 128
    else:
      self.roi_minibatch_size = 64

    self.roi_positive_rate = 0.25
    self.roi_iou_upper_th = 0.5
    self.roi_iou_lower_th = 0.5

    self.roi_encoder_scales = [10., 10., 5., 5.]
    unmatched_cls_target = tf.constant([1] + self.num_things_classes * [0], dtype=tf.float32)
    self.roi_box_matcher = box_matcher.BoxMatcher(self.roi_iou_upper_th,
                                                  self.roi_iou_lower_th,
                                                  unmatched_cls_target=unmatched_cls_target,
                                                  encoder_scales=self.roi_encoder_scales,
                                                  return_iids=True,
                                                  unmatched_iid_target=[-1])

    self.roi_sampler = sampler.ROISampler(params=self.params,
                                          positive_rate=self.roi_positive_rate)

    self.mask_box_matcher = box_matcher.BoxMatcher(self.roi_iou_upper_th,
                                                   self.roi_iou_lower_th,
                                                   unmatched_cls_target=tf.zeros((self.params.height_input,
                                                                                  self.params.width_input)))

    self.mask_size = [28, 28]
    self.mask_num_layers = 2


  def predict(self, features, prediction_dict, groundtruth_dict=None):
    prediction_dict.update(self._predict_rpn(features))

    if self.IMPLICIT_INFO_EXCHANGE:
      with tf.variable_scope("ImplicitInfoExchange"):
        seg_logits = prediction_dict['logits']
        with tf.control_dependencies([seg_logits]):
          seg_logits = tf.image.resize_nearest_neighbor(seg_logits,
                                                        size=tf.shape(features)[1:3],
                                                        align_corners=True)

          seg_logits = tf.nn.l2_normalize(seg_logits, axis=[1, 2, 3])
          features = tf.nn.l2_normalize(features, axis=[1, 2])

          merged_map = tf.concat([features, seg_logits], axis=3)

          with slim.arg_scope(norm_arg_scope(self.params)):
            features = slim.conv2d(
              merged_map,
              512,
              kernel_size=[3, 3],
              rate=1)

    if self.PREDICT_DETECTION:
      prediction_dict = self._predict_detection(prediction_dict, features, groundtruth_dict)

    return prediction_dict

  def _predict_rpn(self, features):
    with tf.variable_scope("RPN"):
      anchors = anchor_generator.generate(base_size=self._anchor_size,
                                          stride=self._anchor_stride,
                                          scales=self._anchor_scales,
                                          ratios=self._anchor_ratios,
                                          features_height=tf.shape(features)[1],
                                          features_width=tf.shape(features)[2],
                                          offset=self._anchor_offset)

      anchors_normalized = box_utils.normalize_boxes(anchors, self.params.height_input, self.params.width_input)

      rpn_sliding_window = slim.conv2d(
        features,
        512,
        kernel_size=[3, 3],
        activation_fn=tf.nn.relu,
        scope='sliding_window')

      rpn_objectness = slim.conv2d(rpn_sliding_window,
                                   self.num_anchors_per_location*2,
                                   kernel_size = [1, 1],
                                   activation_fn = None,
                                   padding = "VALID",
                                   scope='objectness'
                                   )

      rpn_box_encoded = slim.conv2d(rpn_sliding_window,
                            self.num_anchors_per_location*4,
                            kernel_size = [1, 1],
                            activation_fn = None,
                            padding = "VALID",
                            scope='box'
                            )

      rpn_objectness = tf.reshape(rpn_objectness, [self.params.Nb, -1, 2])
      rpn_box_encoded = tf.reshape(rpn_box_encoded, [self.params.Nb, -1, 4])

      prediction_dict = {'rpn_objectness': rpn_objectness,
                         'rpn_box_encoded': rpn_box_encoded,
                         'anchors': anchors,
                         'anchors_normalized': anchors_normalized}

      print(prediction_dict)

    return prediction_dict

  def _predict_detection(self, prediction_dict, features, groundtruth_dict=None):
    with tf.variable_scope("PredictDetection"):
      # Postprocess RPN
      prediction_dict = self._postprocess_rpn(prediction_dict)

      roi_boxes = prediction_dict['rpn_boxes_postprocessed']

      if self.ADD_REGION_PROPOSALS:
        with tf.variable_scope("AddRegionProposals"):
          additional_boxes = explicit_info_exchange.retrieve_rpn_additions(prediction_dict['logits'],
                                                                           100,
                                                                           self.params)

          additional_boxes = tf.stop_gradient(additional_boxes)
          roi_boxes = tf.concat([roi_boxes, additional_boxes], axis=1)
          roi_boxes = tf.stop_gradient(roi_boxes)

      _normalize_fn = functools.partial(box_utils.normalize_boxes,
                                        orig_height=self.params.height_input,
                                        orig_width=self.params.width_input)
      roi_boxes_normalized = tf.map_fn(_normalize_fn, roi_boxes)

      # Sample ROIs
      if self.is_training:
        gt_box_wrappers = groundtruth_dict['box_wrappers']
        roi_boxes_normalized = tf.stop_gradient(roi_boxes_normalized)
        roi_boxes_normalized, matches = self.roi_sampler.batch_sample_and_match(roi_boxes_normalized,
                                                                                gt_box_wrappers,
                                                                                self.roi_minibatch_size,
                                                                                self.roi_box_matcher)
        prediction_dict['matches'] = matches


      # ROI Pool/Align
      roi_features = roi_pooling.batch_roi_pooling(features,
                                                   roi_boxes_normalized,
                                                   self.params)


      # Apply ResNet head
      roi_features_detection = feature_extractor.resnet_faster_rcnn_head(roi_features,
                                                                    self.params)

      roi_features_flat = tf.reduce_mean(roi_features_detection,
                                         axis=[1, 2],
                                         keepdims=False,
                                         name='global_avg_pooling')

      # Apply prediction
      # Predict classification
      det_cls = slim.fully_connected(roi_features_flat,
                                     num_outputs=self.num_things_classes+1,
                                     activation_fn=None,
                                     scope='det_cls'
                                     )
      det_cls = tf.reshape(det_cls, [self.params.Nb, -1, self.num_things_classes+1])

      # Predict regression
      det_reg = slim.fully_connected(roi_features_flat,
                                     num_outputs=(self.num_things_classes+1)*4,
                                     activation_fn=None,
                                     scope='det_reg'
                                     )

      det_reg = tf.reshape(det_reg, [self.params.Nb, -1, self.num_things_classes+1, 4])

      prediction_dict['detection_class_scores'] = det_cls
      prediction_dict['detection_boxes_encoded'] = det_reg
      prediction_dict['rois_sampled_normalized'] = roi_boxes_normalized

    if self.MASK_PREDICTION:
      if self.is_training:
        features_mask = roi_features_detection
      else:
        prediction_dict = self._postprocess_detection(prediction_dict)
        det_boxes_normalized = prediction_dict['det_boxes_normalized']
        if self.EXPAND_BOXES:
          with tf.variable_scope("ExpandBoxes"):
            detection_boxes = prediction_dict['det_boxes_postprocessed']
            detection_classes = prediction_dict['det_class_postprocessed']
            seg_logits = prediction_dict['logits']

            detection_boxes = explicit_info_exchange.expand_instance_bboxes(detection_boxes,
                                                                            detection_classes,
                                                                            seg_logits,
                                                                            self.params)

            _normalize_fn = functools.partial(box_utils.normalize_boxes,
                                              orig_height=self.params.height_input,
                                              orig_width=self.params.width_input)
            det_boxes_normalized = tf.map_fn(_normalize_fn, detection_boxes)
            prediction_dict['det_boxes_normalized'] = det_boxes_normalized

        with tf.variable_scope("PredictDetection"):
          features_mask = roi_pooling.batch_roi_pooling(features,
                                                        det_boxes_normalized,
                                                        self.params)
          features_mask = feature_extractor.resnet_faster_rcnn_head(features_mask,
                                                                    self.params)

      prediction_dict.update(self._predict_masks(features_mask))

    return prediction_dict

  def _predict_masks(self, features):
    prediction_dict = dict()
    with tf.variable_scope("MaskPrediction"):
      with slim.arg_scope(norm_arg_scope(self.params)):
        if self.params.mask_arch == 'transpose':
          features = slim.conv2d_transpose(features,
                                           256,
                                           [2, 2],
                                           stride=2,
                                           normalizer_fn=slim.batch_norm)

          for _ in range(self.params.mask_num_layers - 2):
            features = slim.conv2d(features, 256, [3, 3],
                                   normalizer_fn=slim.batch_norm)

          features = slim.conv2d_transpose(features,
                                           256,
                                           [2, 2],
                                           stride=2,
                                           normalizer_fn=slim.batch_norm)

        else:
          features = slim.conv2d(features, 256, [3, 3],
                                 normalizer_fn=slim.batch_norm)

          features = tf.image.resize_bilinear(features,
                                              self.mask_size,
                                              align_corners=True)

          for _ in range(self.params.mask_num_layers - 1):
            features = slim.conv2d(features, 256, [3, 3],
                                   normalizer_fn=slim.batch_norm)

      mask_logits = slim.conv2d(features,
                                  self.num_things_classes,
                                  [1, 1],
                                  normalizer_fn=None,
                                  activation_fn=None)

      mask_logits = tf.reshape(mask_logits, [self.params.Nb, -1, self.mask_size[0], self.mask_size[1], self.num_things_classes])
      mask_logits = tf.transpose(mask_logits, [0, 1, 4, 2, 3])

      prediction_dict['mask_logits'] = mask_logits

    return prediction_dict

  def loss(self, prediction_dict, groundtruth_dict):
    loss_dict = dict()
    loss_dict.update(self._loss_rpn(prediction_dict, groundtruth_dict))

    if self.PREDICT_DETECTION:
      loss_dict.update(self._loss_detection(prediction_dict, groundtruth_dict))

    return loss_dict

  def _loss_rpn(self, prediction_dict, groundtruth_dict):
    with tf.variable_scope("LossRPN"):
      rpn_objectness = prediction_dict['rpn_objectness']
      rpn_box_encoded = prediction_dict['rpn_box_encoded']
      anchors_norm = prediction_dict['anchors_normalized']

      gt_box_wrappers = groundtruth_dict['box_wrappers']

      # Match anchors to ground truth boxes and find targets, weights for cls, box loss
      box_targets, cls_targets, box_weights, cls_weights = box_matcher.batch_match_boxes(self.rpn_box_matcher,
                                                                                         gt_box_wrappers,
                                                                                         anchors_norm)


      # Subsample a minibatch for the loss
      def _sample_fn(inputs):
        with tf.variable_scope("SampleLossMinibatch"):
          cls_t, cls_w = inputs
          cls_w = tf.cast(tf.reshape(cls_w, [-1]), tf.bool)
          cls_t = tf.cast(tf.reshape(cls_t, [-1]), tf.bool)
          sampled_indices = self.rpn_sampler.sample(tf.reshape(cls_w, [-1]),
                                                    self.rpn_minibatch_size,
                                                    tf.reshape(cls_t, [-1]))
        return tf.cast(sampled_indices, tf.float32)

      batch_sampled_indices = tf.map_fn(_sample_fn,
                                        [cls_targets, cls_weights],
                                        tf.float32)

      batch_box_targets = tf.reshape(box_targets, [self.params.Nb, -1, 4])
      batch_cls_targets = tf.reshape(cls_targets, [self.params.Nb, -1])
      batch_box_weights = tf.reshape(box_weights, [self.params.Nb, -1])

      # Calculate the box regression loss
      sampled_box_weights = batch_sampled_indices * batch_box_weights
      rpn_reg_losses = losses.rpn_regression_loss(rpn_box_encoded,
                                                  batch_box_targets,
                                                  weights=sampled_box_weights)

      normalizer = tf.reduce_sum(batch_sampled_indices, axis=1)

      rpn_reg_loss = tf.reduce_sum(tf.reduce_sum(rpn_reg_losses, axis=1))
      rpn_reg_loss = rpn_reg_loss * self.rpn_regression_loss_weight / tf.reduce_sum(normalizer)

      # Calculate the classification loss
      batch_cls_targets_oh = tf.one_hot(tf.to_int32(batch_cls_targets), depth=2)
      rpn_obj_losses = losses.rpn_objectness_loss(rpn_objectness,
                                                  batch_cls_targets_oh,
                                                  weights=batch_sampled_indices)
      rpn_obj_loss = tf.reduce_mean(tf.reduce_sum(rpn_obj_losses, axis=1) / normalizer)
      rpn_obj_loss = rpn_obj_loss * self.rpn_classification_loss_weight

    tf.summary.scalar('rpn_reg', rpn_reg_loss, family='losses')
    tf.summary.scalar('rpn_obj', rpn_obj_loss, family='losses')

    loss_dict = {'rpn_reg': rpn_reg_loss,
                 'rpn_obj': rpn_obj_loss}

    return loss_dict

  def _loss_detection(self, prediction_dict, groundtruth_dict):
    with tf.variable_scope("LossDetection"):

      loss_dict = dict()

      matches = prediction_dict['matches']
      box_targets = matches['box_targets']
      cls_targets = matches['cls_targets']
      box_weights = matches['box_weights']
      cls_weights = matches['cls_weights']

      class_scores = prediction_dict['detection_class_scores']
      boxes_encoded = prediction_dict['detection_boxes_encoded']
      boxes_encoded_reshaped = tf.reshape(boxes_encoded, [self.params.Nb, -1, self.num_things_classes+1, 4])

      boxes_encoded_for_matched_classes = tf.boolean_mask(
        boxes_encoded_reshaped,
        cls_targets)

      normalizer = self.params.Nb * self.roi_minibatch_size
      box_normalizer = tf.maximum(tf.reduce_sum(tf.reduce_sum(box_weights)), 1.)

      if not self.params.use_box_normalizer:
        box_normalizer = normalizer

      # Calculate regression loss
      boxes_encoded_reshaped = tf.reshape(boxes_encoded_for_matched_classes, [self.params.Nb, -1, 4])
      box_weights = tf.reshape(box_weights, [self.params.Nb, -1])
      detection_box_losses = losses.detection_regression_loss(boxes_encoded_reshaped,
                                                              box_targets,
                                                              weights=box_weights)
      detection_box_loss = tf.reduce_sum(detection_box_losses) / tf.cast(box_normalizer, tf.float32)
      detection_box_loss = detection_box_loss * self.params.detection_box_loss_weight

      detection_cls_losses = losses.detection_classification_loss(class_scores,
                                                                  cls_targets,
                                                                  weights=cls_weights)
      detection_cls_loss = tf.reduce_sum(detection_cls_losses) / tf.cast(normalizer, tf.float32)
      detection_cls_loss = detection_cls_loss * self.params.detection_cls_loss_weight

    tf.summary.scalar('det_box', detection_box_loss, family='losses')
    tf.summary.scalar('det_cls', detection_cls_loss, family='losses')

    loss_dict['det_box'] = detection_box_loss
    loss_dict['det_cls'] = detection_cls_loss

    if self.MASK_PREDICTION:
      loss_dict.update(self._loss_masks(prediction_dict, groundtruth_dict))

    return loss_dict

  def _loss_masks(self, prediction_dict, groundtruth_dict):
    with tf.variable_scope("LossMasks"):
      loss_dict = dict()

      # Match boxes to groundtruth and retrieve unique ID
      matches = prediction_dict['matches']
      iid_targets = matches['iid_targets']
      cls_targets = matches['cls_targets']
      mask_weights = matches['box_weights']

      gt_masks = groundtruth_dict['instance_masks']
      roi_boxes = prediction_dict['rois_sampled_normalized']

      # Extract masks based on unique ID
      mask_targets = mask_utils.extract_masks_by_id(gt_masks,
                                                    iid_targets,
                                                    roi_boxes,
                                                    self.params,
                                                    crop_size=self.mask_size)
      mask_targets = tf.stop_gradient(mask_targets)

      mask_targets_flat = tf.reshape(mask_targets,
                                      [self.params.Nb, -1,
                                       self.mask_size[0] * self.mask_size[1]])

      # Pad the prediction and retrieve the mask with the predicted class
      mask_logits = prediction_dict['mask_logits']

      mask_logits = tf.reshape(mask_logits, [-1, self.num_things_classes, self.mask_size[0], self.mask_size[1]])
      mask_logits_with_background = tf.pad(
        mask_logits, [[0, 0], [1, 0], [0, 0], [0, 0]])
      mask_logits_with_background_reshaped = tf.reshape(mask_logits_with_background,
                                                        [self.params.Nb,
                                                         -1,
                                                         self.num_things_classes+1,
                                                         self.mask_size[0],
                                                         self.mask_size[1]])

      mask_logits_per_class = tf.boolean_mask(mask_logits_with_background_reshaped,
                                              cls_targets)
      mask_logits_per_class = tf.reshape(mask_logits_per_class,
                                                        [self.params.Nb,
                                                         -1,
                                                         self.mask_size[0],
                                                         self.mask_size[1]])
      mask_logits_flat = tf.reshape(mask_logits_per_class,
                                     [self.params.Nb, -1,
                                      self.mask_size[0] * self.mask_size[1]])

      mask_weights = tf.reshape(mask_weights, [self.params.Nb, -1])

      # Apply sigmoid cross entropy loss
      mask_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_targets_flat,
                                                          logits=mask_logits_flat)
      mask_losses = tf.reduce_sum(mask_losses, axis=2)
      mask_losses = mask_losses * mask_weights

      mask_normalizer = self.mask_size[0] * self.mask_size[1] * tf.maximum(
            tf.reduce_sum(mask_weights, axis=1, keepdims=True),
            tf.ones((self.params.Nb, 1)))

      mask_loss = tf.reduce_sum(mask_losses / mask_normalizer)
      mask_loss = mask_loss * self.params.mask_loss_weight

    self.visualize_masks(mask_logits_per_class,
                         mask_targets,
                         prediction_dict['detection_class_scores'])

    tf.summary.scalar('mask', mask_loss, family='losses')

    loss_dict['mask'] = mask_loss

    return loss_dict

  def postprocess(self, prediction_dict):
    if self.is_training:
      prediction_dict = self._postprocess_detection(prediction_dict)
    else:
      prediction_dict = self._postprocess_masks(prediction_dict)

    return prediction_dict

  def _postprocess_rpn(self,
                       prediction_dict):
    with tf.variable_scope("PostprocessRPN"):

      img_shape = [self.params.height_input, self.params.width_input]
      def _decode_and_nms_fn(inputs, anchors):
        with tf.variable_scope("DecodeAndApplyNMS"):
          boxes_encoded = inputs[0]
          boxes_scores = inputs[1]
          boxes_decoded = box_utils.decode_boxes(boxes_encoded, anchors)
          boxes_resized = box_utils.resize_normalized_boxes(boxes_decoded, img_shape[0], img_shape[1])
          boxes_clipped = box_utils.clip_to_img_boundaries(boxes_resized,
                                                           image_shape=img_shape)

          boxes_probs = slim.softmax(boxes_scores)

          boxes_clipped_formatted = box_utils.convert_xyxy_to_yxyx_format(boxes_clipped)

          keep_boxes_ids = tf.image.non_max_suppression(
            boxes=boxes_clipped_formatted,
            scores=boxes_probs[:, 1],
            max_output_size=self.rpn_nms_num_samples,
            iou_threshold=self.rpn_nms_iou_th
          )

          boxes_out = tf.gather(boxes_clipped, keep_boxes_ids)
          probs_out = tf.gather(boxes_probs, keep_boxes_ids)

          return boxes_out, probs_out

      boxes_encoded = prediction_dict['rpn_box_encoded']
      boxes_scores = prediction_dict['rpn_objectness']
      anchors_norm = prediction_dict['anchors_normalized']

      _decode_and_nms_fn = functools.partial(_decode_and_nms_fn, anchors=anchors_norm)
      boxes_postproc, probs_postproc = tf.map_fn(_decode_and_nms_fn,
                                                  [boxes_encoded, boxes_scores],
                                                dtype=(tf.float32, tf.float32))

    prediction_dict['rpn_boxes_postprocessed'] = boxes_postproc
    prediction_dict['rpn_probs_postprocessed'] = probs_postproc

    return prediction_dict

  def _postprocess_detection(self,
                             prediction_dict):
    with tf.variable_scope("PostprocessDetection"):
      img_shape = [self.params.height_input, self.params.width_input]
      def _decode_and_nms_fn(inputs):
        with tf.variable_scope("DecodeAndApplyNMS"):
          boxes_encoded = inputs[0]
          boxes_scores = inputs[1]
          rois = inputs[2]

          boxes_probs = slim.softmax(boxes_scores)
          boxes_classes = tf.argmax(boxes_probs, axis=1)
          # Do not include background prediction
          boxes_probs_red = tf.reduce_max(boxes_probs[..., 1:], axis=1)
          boxes_classes_one_hot = tf.cast(tf.one_hot(boxes_classes,
                                                   depth=self.params.num_things_classes+1), tf.bool)

          pad_num = tf.shape(boxes_classes)[0]

          boxes_encoded_per_class = tf.boolean_mask(boxes_encoded, boxes_classes_one_hot)
          boxes_encoded_per_class = tf.reshape(boxes_encoded_per_class, [-1, 4])

          # Decode boxes
          boxes_decoded = box_utils.decode_boxes(boxes_encoded_per_class,
                                                 rois,
                                                 scale_factors=self.roi_encoder_scales)

          # Clip boxes to image boundaries
          boxes_resized = box_utils.resize_normalized_boxes(boxes_decoded, img_shape[0], img_shape[1])
          boxes_clipped = box_utils.clip_to_img_boundaries(boxes_resized,
                                                           image_shape=img_shape)

          # Find indices of boxes with score above the threshold and gather
          indices = tf.reshape(tf.where(tf.greater(boxes_probs_red, self.params.det_nms_score_th)), [-1])
          boxes_clipped = tf.gather(boxes_clipped, indices)
          boxes_probs_red = tf.gather(boxes_probs_red, indices)

          # Subtract the background class from the predicted classes
          boxes_classes = tf.gather(boxes_classes, indices) - 1

          boxes_clipped_formatted = box_utils.convert_xyxy_to_yxyx_format(boxes_clipped)

          keep_boxes_ids = tf.image.non_max_suppression(
            boxes=boxes_clipped_formatted,
            scores=boxes_probs_red,
            max_output_size=pad_num,
            iou_threshold = self.params.det_nms_iou_th
          )

          boxes_out = tf.gather(boxes_clipped, keep_boxes_ids)
          probs_out = tf.gather(boxes_probs_red, keep_boxes_ids)
          class_out = tf.gather(boxes_classes, keep_boxes_ids)

          boxes_pad, num_boxes = box_utils.pad_boxes_and_return_num(boxes_out, pad_num)
          probs_pad = tf.pad(probs_out, [[0, pad_num - num_boxes]])
          class_pad = tf.pad(class_out, [[0, pad_num - num_boxes]])

          boxes_pad = tf.reshape(boxes_pad, [pad_num, 4])
          probs_pad = tf.reshape(probs_pad, [pad_num])
          class_pad = tf.reshape(class_pad, [pad_num])

          return boxes_pad, class_pad, probs_pad, num_boxes

      boxes_encoded = prediction_dict['detection_boxes_encoded']
      boxes_scores = prediction_dict['detection_class_scores']
      rois_norm = prediction_dict['rois_sampled_normalized']

      boxes_postproc, class_postproc, probs_postproc, num_boxes = tf.map_fn(_decode_and_nms_fn,
                                                  [boxes_encoded, boxes_scores, rois_norm],
                                                  dtype=(tf.float32, tf.int64, tf.float32, tf.int32))

    prediction_dict['det_boxes_postprocessed'] = boxes_postproc
    prediction_dict['det_class_postprocessed'] = class_postproc
    prediction_dict['det_probs_postprocessed'] = probs_postproc
    prediction_dict['det_num_boxes'] = num_boxes

    _normalize_fn = functools.partial(box_utils.normalize_boxes,
                                      orig_height=self.params.height_input,
                                      orig_width=self.params.width_input)
    det_boxes_normalized = tf.map_fn(_normalize_fn, boxes_postproc)
    prediction_dict['det_boxes_normalized'] = det_boxes_normalized

    return prediction_dict

  def _postprocess_masks(self,
                         prediction_dict):
    with tf.variable_scope("PostprocessMasks"):
      batch_num_masks = prediction_dict['det_num_boxes']
      prediction_dict['masks_sigmoid'] = tf.nn.sigmoid(prediction_dict['mask_logits'])
      batch_class_predictions = prediction_dict['det_class_postprocessed']
      batch_normalized_boxes = prediction_dict['det_boxes_normalized']

      batch_num_masks = tf.unstack(batch_num_masks, num=self.params.Nb)
      batch_masks_sigmoid = tf.unstack(prediction_dict['masks_sigmoid'], num=self.params.Nb)
      batch_class_predictions = tf.unstack(batch_class_predictions, num=self.params.Nb)
      batch_normalized_boxes = tf.unstack(batch_normalized_boxes, num=self.params.Nb)

      batch_masks_reshaped = list()
      batch_masks_per_class = list()
      batch_masks_reshaped_probs = list()

      for masks_sigmoid, class_predictions, normalized_boxes, num_masks in zip(
              batch_masks_sigmoid, batch_class_predictions, batch_normalized_boxes, batch_num_masks):

        masks_sigmoid = masks_sigmoid[:num_masks]
        class_predictions = class_predictions[:num_masks]
        normalized_boxes = normalized_boxes[:num_masks]

        class_one_hot = tf.cast(tf.one_hot(class_predictions, depth=self.num_things_classes), tf.bool)
        masks_per_class = tf.boolean_mask(masks_sigmoid, class_one_hot)

        masks_reshaped_probs = mask_utils.reframe_box_masks_to_image_masks(
          masks_per_class, normalized_boxes, self.params.height_input, self.params.width_input)
        masks_reshaped = tf.cast(tf.greater_equal(masks_reshaped_probs, 0.5), tf.uint8)

        batch_masks_per_class.append(masks_per_class)
        batch_masks_reshaped.append(masks_reshaped)
        batch_masks_reshaped_probs.append(masks_reshaped_probs)

      prediction_dict['masks_per_class'] = tf.stack(batch_masks_per_class)
      prediction_dict['masks_reshaped'] = tf.stack(batch_masks_reshaped)
      prediction_dict['masks_reshaped_probs'] = tf.stack(batch_masks_reshaped_probs)

      return prediction_dict

  def visualize_masks(self, mask_logits, mask_targets, scores):
    with tf.variable_scope("VisualizeMasks"):
      masks_sigmoid = tf.nn.sigmoid(mask_logits)
      mask_pred_th = tf.cast(tf.greater_equal(masks_sigmoid, 0.5), tf.uint8)
      mask_pred_img = mask_pred_th * 255

      mask_targets_img = tf.cast(mask_targets * 255, tf.uint8)

      def _draw_fn(inputs):
        mask_pred_img = inputs[0]
        mask_targets_img = inputs[1]
        scores = inputs[2]

        boxes_probs = slim.softmax(scores)
        boxes_probs_red = tf.reduce_max(boxes_probs[..., 1:], axis=1)

        _, indices = tf.nn.top_k(boxes_probs_red, k=3)
        mask_pred_img = tf.gather(mask_pred_img, indices)
        mask_targets_img = tf.gather(mask_targets_img, indices)

        return [mask_pred_img, mask_targets_img]

      imgs = tf.map_fn(_draw_fn,
                       [mask_pred_img, mask_targets_img, scores],
                       dtype=[tf.uint8, tf.uint8])

      mask_pred_img = tf.reshape(imgs[0], [-1, self.mask_size[0], self.mask_size[1], 1])
      mask_targets_img = tf.reshape(imgs[1], [-1, self.mask_size[0], self.mask_size[1], 1])

    tf.summary.image('mask_pred', mask_pred_img, max_outputs=6)
    tf.summary.image('mask_gt', mask_targets_img, max_outputs=6)

  def format_gt_dict(self, gt_dict_input):
    with tf.variable_scope("FormatGroundTruthDict"):
      boxes_batch = tf.unstack(gt_dict_input['boxes'], num=self.params.Nb)
      classes_batch = tf.unstack(gt_dict_input['classes'], num=self.params.Nb)
      weights_batch = tf.unstack(gt_dict_input['weights'], num=self.params.Nb)
      num_boxes_batch = tf.unstack(gt_dict_input['num_boxes'], num=self.params.Nb)
      instance_ids_batch = tf.unstack(gt_dict_input['instance_ids'], num=self.params.Nb)

      def _format_classes(classes, params):
        cond = tf.reshape(tf.equal(classes[0], -1), [])

        classes_ids = tf.cond(cond,
                              lambda: classes,
                              lambda: tf.gather(params.cids2object_ids, classes))

        return classes_ids

      gt_box_wrappers = list()
      for boxes, classes, weights, num_boxes, instance_ids in \
              zip(boxes_batch, classes_batch, weights_batch, num_boxes_batch, instance_ids_batch):
        gt_box_wrapper = box_wrapper.init_box_wrappers_from_dataset(boxes, classes, weights, num_boxes, instance_ids)
        gt_box_wrapper.set_classes(_format_classes(gt_box_wrapper.get_classes(), self.params))
        gt_box_wrappers.append(gt_box_wrapper)

    groundtruth_dict = {'box_wrappers': gt_box_wrappers,
                        'instance_masks': gt_dict_input['instance_masks']}

    return groundtruth_dict

def norm_arg_scope(params):
  batch_norm_params = {
    'decay': params.batch_norm_decay,
    'epsilon': 1e-5,
    'scale': True,
    'trainable': True,
    'is_training': params.batch_norm_istraining,
    'updates_collections': tf.GraphKeys.UPDATE_OPS}

  with tf.contrib.framework.arg_scope(
      [slim.conv2d],
      # weights_regularizer=slim.l2_regularizer(params.regularization_weight),
      weights_regularizer=None,
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with tf.contrib.framework.arg_scope(
        [slim.batch_norm],
        **batch_norm_params):
      with tf.contrib.framework.arg_scope(
          [slim.max_pool2d],
          padding='SAME') as arg_sc:
        return arg_sc
