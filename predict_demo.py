import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import PanopticSegmentationModel
from config import Params
from PIL import Image
from utils.load_json_to_params import load_json_to_params
import time
import argparse
from skimage import img_as_float
import os
import matplotlib.pyplot as plt
import cv2

OFFSET = 1000
THICK_BORDERS = False

def get_arguments():
  parser = argparse.ArgumentParser(description="Panoptic-Slim Network")
  parser.add_argument("--json_path", type=str, default='examples/params/params_01.json',
                      help="The path to the json file containing the parameters")
  parser.add_argument("--image_dir", type=str, default='examples/demo_images/',
                      help="The path to the directory with images to make predictions on")
  parser.add_argument("--save_predictions", action='store_true',
                      help="Save the panoptic predictions")
  parser.add_argument("--save_dir", type=str, default='output/save_dir/',
                      help="he path to the directory where the panoptic predictions should be saved")

  return parser.parse_args()

def predict(params, filenames_list):
  image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))
  image_ph = tf.reshape(image_ph, [1,
                                   tf.shape(image_ph)[0],
                                   tf.shape(image_ph)[1],
                                   3])

  model = PanopticSegmentationModel(image_ph, None, params)

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

  ckpt = tf.train.get_checkpoint_state(params.log_dir)

  if ckpt and ckpt.model_checkpoint_path:
    loader = tf.train.Saver(var_list=restore_var)
    model.load(loader, sess, ckpt.model_checkpoint_path)
  else:
    print('No checkpoint file found.')

  # # Plotting settings
  # fig = plt.figure(0, [16, 8], dpi=80)
  # ax = plt.Axes(fig, [0., 0., 1., 1.])
  # ax.set_axis_off()
  # fig.add_axes(ax)

  # Plotting settings
  fig = plt.figure(0, [16, 16], dpi=80)
  ax = plt.Axes(fig, [0., 0.5, 1., 0.5])
  ax.set_axis_off()
  fig.add_axes(ax)

  # Plotting settings
  # fig2 = plt.figure(1, [16, 8], dpi=80)
  ax2 = plt.Axes(fig, [0., 0., 1., 0.5])
  ax2.set_axis_off()
  fig.add_axes(ax2)

  for i, filename in enumerate(filenames_list):
    image = Image.open(filename)
    image = image.resize([params.width_input, params.height_input], Image.ANTIALIAS)
    image = np.array(image)
    image = np.array(img_as_float(image)).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # Subtract mean
    image = (image - 0.5) / 0.5

    start_time = time.time()
    print(filename)
    prediction_dict_out, image_out = sess.run([prediction_dict, image_ph],
                                              feed_dict={image_ph: image})

    duration = time.time() - start_time
    print('({:.3f} sec/step)'.format(duration))

    if params.apply_semantic_branch and params.apply_instance_branch:
      panoptic_out = prediction_dict_out['panoptic'][0]
      if not params.save_predictions:
        class_ids = panoptic_out[..., 0]
        max = len(params.cids2colors) - 1
        class_ids[class_ids == 255] = max

        colorpalettes = np.array(params.cids2colors, dtype=np.uint8)
        class_colors = colorpalettes[class_ids]

        panoptic_for_edges = np.stack([class_ids, panoptic_out[..., 1], np.zeros_like(class_ids)], axis=2).astype(np.uint8)

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
        ax2.imshow(img_obj)

        img_obj_2 = Image.open(filename)
        ax.imshow(img_obj_2)

        plt.waitforbuttonpress(timeout=10)

      else:
        class_ids = panoptic_out[..., 0]
        max_cid = len(params.cids2colors) - 1
        class_ids[class_ids == 255] = max_cid

        class_ids2label_ids = np.array(params.cids2labels)
        label_ids = class_ids2label_ids[class_ids]
        instance_ids = panoptic_out[..., 1]
        out_file = np.stack([label_ids, instance_ids], axis=2)

        print(np.min(label_ids), np.max(label_ids))

        save_dir = params.save_dir
        im_name_base = os.path.splitext(os.path.basename(str(filename)))[0]
        out_fname = os.path.join(save_dir, im_name_base + '.png')
        print(out_fname)

        print(np.min(out_file[..., 0]), np.max(out_file[..., 0]))

        Image.fromarray(out_file.astype(np.uint8)).save(out_fname)

if __name__ == '__main__':
  args = get_arguments()
  params = Params()
  params = load_json_to_params(params, args.json_path)
  params.num_steps_predict = params.num_steps_eval
  params.save_predictions = args.save_predictions
  params.save_dir = args.save_dir

  params.is_training = False
  params.batch_norm_istraining = False
  params.height_input = 512
  params.width_input = 1024
  params.height_orig = 604
  params.width_orig = 960
  params.Nb = 1

  filenames_list = list()
  for file in os.listdir(args.image_dir):
    if file.endswith(".png") or file.endswith(".jpg"):
      filenames_list.append(os.path.join(args.image_dir, file))

  predict(params, filenames_list)




