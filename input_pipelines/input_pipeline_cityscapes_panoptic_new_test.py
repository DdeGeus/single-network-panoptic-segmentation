import tensorflow as tf
from input_pipelines.input_pipeline_cityscapes_panoptic_new import train_input
from PIL import Image
import numpy as np
from utils.box_wrapper import BoxWrapper

sess = tf.InteractiveSession()

class params(object):
  height_input = 512
  width_input = 1024
  dataset_directory = '/home/ddegeus/datasets/Cityscapes/training/'
  filelist_filepath = '/home/ddegeus/datasets/Cityscapes/training/panoptic/filenames.lst'
  Nb = 2
  random_flip = True
  random_seed = 1


with tf.device('/cpu:0'):
  dataset = train_input(params)
  features, labels, instance_gt = dataset.make_one_shot_iterator().get_next()

  print(features, labels)

  for i in range(3000):
    fe, la, i_gt = sess.run((features, labels, instance_gt))
    print(fe.shape)
    print(la.shape)
    # print(fe.dtype)
    # print(la.dtype)
    print(i_gt.keys())
    print(i_gt['instance_ids'])
    print(i_gt['classes'])


  fe_img = Image.fromarray(fe[0].astype(np.uint8))
  fe_img.show()
  la_img = Image.fromarray(la[0].astype(np.uint8))
  la_img.show()


