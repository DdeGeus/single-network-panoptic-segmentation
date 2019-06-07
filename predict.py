import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from input_pipelines.input_pipeline_cityscapes import inference_input
from model import PanopticSegmentationModel
from config import Params
from PIL import Image, ImageDraw
from utils import draw_utils, mask_utils_np
from utils.load_json_to_params import load_json_to_params
import time
import argparse
import skimage
import os
from scipy.misc import imresize
import matplotlib.pyplot as plt
import cv2

OFFSET = 1000
THICK_BORDERS = False

def get_arguments():
  parser = argparse.ArgumentParser(description="Panoptic-Slim Network")
  parser.add_argument("--json_path", type=str, default='',
                      help="The path to the json file containing the parameters")

  return parser.parse_args()

def predict(params):
  dataset = inference_input(params)
  iterator = dataset.make_one_shot_iterator()

  image, im_name, im_raw = iterator.get_next()
  model = PanopticSegmentationModel(image, None, params)

  prediction_dict = model.predict()
  prediction_dict = model.postprocess(prediction_dict)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)
  global_init = tf.global_variables_initializer()
  local_init = tf.local_variables_initializer()
  sess.run(global_init)
  sess.run(local_init)

  restore_var = tf.global_variables()

  ckpt = tf.train.get_checkpoint_state(params.checkpoint_dir)

  if ckpt and ckpt.model_checkpoint_path:
    loader = tf.train.Saver(var_list=restore_var)
    model.load(loader, sess, ckpt.model_checkpoint_path)
  else:
    print('No checkpoint file found.')

  # Plotting settings
  fig = plt.figure(0, [16, 8], dpi=80)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                          output_partition_graphs=True)
  run_metadata = tf.RunMetadata()

  for i in range(params.num_steps_predict):
    start_time = time.time()
    print(i)
    if i == 1:
      prediction_dict_out, image_out, im_name_out, im_raw_out = sess.run([prediction_dict, image, im_name, im_raw],
                                                                         options=options,
                                                                         run_metadata=run_metadata)
      fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      chrome_trace = fetched_timeline.generate_chrome_trace_format()
      new_path = os.path.join(params.log_dir, 'timeline_predict.json')
      print(new_path)
      with open(new_path, 'w') as f:
        f.write(chrome_trace)

    else:
      prediction_dict_out, image_out, im_name_out, im_raw_out = sess.run([prediction_dict, image, im_name, im_raw])

    duration = time.time() - start_time
    print('({:.3f} sec/step)'.format(duration))

    if params.apply_semantic_branch and params.apply_instance_branch:
      panoptic_out = prediction_dict_out['panoptic'][0]

      class_ids = panoptic_out[..., 0]
      max = len(params.cids2colors) - 1
      class_ids[class_ids == 255] = max
      # class_ids[class_ids > max] = max

      colorpalettes = np.array(params.cids2colors, dtype=np.uint8)
      class_colors = colorpalettes[class_ids]

      panoptic_for_edges = np.stack([class_ids, panoptic_out[..., 1], np.zeros_like(class_ids)], axis=2).astype(np.uint8)
      # print(panoptic_for_edges.shape)

      edges = cv2.Canny(panoptic_for_edges, 1, 2)

      if THICK_BORDERS:
        edges_2 = cv2.Canny(edges, 100, 200)
        edges_total = np.minimum(edges + edges_2, 255)
      else:
        edges_total = edges

      edges_bool = (edges_total / 255).astype(np.bool)
      edges_invert = np.invert(edges_bool)
      edges_invert = edges_invert.astype(np.uint8)

      class_colors = class_colors.astype(np.uint8) * np.expand_dims(edges_invert, axis=2) + np.expand_dims(
        edges_total, axis=2)

      img_obj = Image.fromarray(np.uint8(class_colors))

      ax.imshow(img_obj)
      plt.waitforbuttonpress(timeout=5)




if __name__ == '__main__':
  args = get_arguments()
  params = Params()
  params = load_json_to_params(params, args.json_path)
  params.dataset_directory = '/home/ddegeus/datasets/Cityscapes/validation/'
  params.filelist_filepath = '/home/ddegeus/datasets/Cityscapes/validation/panoptic/filenames.lst'
  params.is_training = False
  params.batch_norm_istraining = False
  params.num_steps_predict = params.num_steps_eval
  params.height_input = 512
  params.width_input = 1024
  params.Nb = 1

  predict(params)




