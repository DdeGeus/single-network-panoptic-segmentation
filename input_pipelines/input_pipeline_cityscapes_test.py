import tensorflow as tf
from input_pipelines.input_pipeline_cityscapes import train_input
from PIL import Image
import numpy as np

sess = tf.InteractiveSession()

class params(object):
  height_input = 512
  width_input = 1024
  dataset_directory = '/home/ddegeus/datasets/Cityscapes/training/'
  filelist_filepath = '/home/ddegeus/datasets/Cityscapes/training/panoptic/filenames.lst'
  Nb = 3


with tf.device('/cpu:0'):
  dataset = train_input(params)
  features, labels = dataset.make_one_shot_iterator().get_next()

  print(features, labels)

  for i in range(3000):
    fe, la = sess.run((features, labels))
    print(fe.shape)
    print(la.shape)
    print(fe.dtype)
    print(la.dtype)

  fe_img = Image.fromarray(fe[0].astype(np.uint8))
  fe_img.show()
  la_img = Image.fromarray(la[0].astype(np.uint8))
  la_img.show()


