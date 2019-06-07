class Params(object):
  height_input = 512
  width_input = 1024
  height_orig = 1024
  width_orig = 2048
  dataset_directory = '/home/ddegeus/datasets/Cityscapes/validation/'
  filelist_filepath = '/home/ddegeus/datasets/Cityscapes/validation/panoptic/filenames.lst'
  init_ckpt_path = '/home/ddegeus/additional_data/pretrained/resnet_v1_50.ckpt'
  Nb = 1
  regularization_weight = 0.000001
  batch_norm_decay = 0.9
  batch_norm_istraining = False
  is_training = False

  learning_rate = 0.01
  lr_schedule = 'poly'
  lr_power = 0.9
  lr_boundaries = [20001, 35001]
  lr_step_factor = 0.2

  momentum = 0.9

  num_steps = 25001
  num_steps_eval = 500
  train_beta_gamma = True
  ckpt_save_steps = 10000
  save_summaries = 200

  base_anchor = 128

  rpn_regression_loss_weight = 1.0
  rpn_classification_loss_weight = 1.0
  use_box_normalizer = False
  detection_box_loss_weight = 1.0
  detection_cls_loss_weight = 1.0
  mask_loss_weight = 1.0

  mask_arch = 'upsample'
  mask_num_layers = 2

  semantic_segmentation_loss_weight = 1.0

  log_dir = 'examples/checkpoint/'
  predict_dir = '/home/ddegeus/datasets/Cityscapes/validation/images/'
  results_dir = '/home/ddegeus/hdd/results_dir/test/'
  checkpoint_dir = 'examples/checkpoint/'


  labels2colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128),
                   (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                   (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                   (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
                   (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230),
                   (119, 11, 32), (0, 0, 142)]
  cids2colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                 [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
                 [0, 0, 0]]
  labels2cids = [-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, -1, -1, 16, 17, 18]
  cids2labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 255]
  num_things_classes = 8

  cids2object_ids = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, -1, -1, 5, 6, 7]
  object_cids2lids = [24, 25, 26, 27, 28, 31, 32, 33]
  cids2is_object = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  object_cids2labels = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
  num_classes = max(labels2cids) + 1

  roi_crop_size = [14, 14]
  roi_pool_kernel_size = 2

  det_nms_iou_th = 0.4
  det_nms_score_th = 0.5

  psp_module = True
  hybrid_upsampling = True
  resnet_init = True
  random_flip = False
  apply_semantic_branch = True
  apply_instance_branch = True
  random_seed = 1

  show_eval_img = True

  rpn_roi_settings = 'old'

