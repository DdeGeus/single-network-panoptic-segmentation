import tensorflow as tf
import functools
import os
import glob
import numpy as np
from PIL import Image

def from_0_1_to_m1_1(images):
  """
  Center images from [0, 1) to [-1, 1).

  Arguments:
    images: tf.float32, in [0, 1), of any dimensions

  Return:
    images linearly scaled to [-1, 1)
  """

  # shifting from [0, 1) to [-1, 1) is equivalent to assuming 0.5 mean
  mean = 0.5
  proimages = (images - mean) / mean

  return proimages

def _parse_and_decode(filename, dataset_directory):
  """

  Args:
    filename:
    dataset_directory:

  Returns:

  """
  filename_split = tf.unstack(tf.string_split([filename], "_").values[:-1], num=3)
  strip_filename = tf.string_join(filename_split, "_")

  im_dir = tf.cast(os.path.join(dataset_directory, 'images/'), tf.string)
  la_dir = tf.cast(os.path.join(dataset_directory, 'panoptic_proc/'), tf.string)
  im_ext = tf.cast('.png', tf.string)
  la_ext = tf.cast('_gtFine_instanceIds.png', tf.string)

  im_filename = tf.string_join([im_dir, filename, im_ext])
  la_filename = tf.string_join([la_dir, strip_filename, la_ext])

  im_dec = tf.image.decode_jpeg(tf.read_file(im_filename))

  # Check if the image is in greyscale and convert to RGB if so
  greyscale_cond = tf.equal(tf.shape(im_dec)[-1], 1)
  im_dec = tf.cond(greyscale_cond,
                   lambda: tf.image.grayscale_to_rgb(im_dec),
                   lambda: tf.identity(im_dec))

  im_dec = tf.image.convert_image_dtype(im_dec, dtype=tf.float32)
  im_dec = from_0_1_to_m1_1(im_dec)

  la_dec = tf.image.decode_png(tf.read_file(la_filename))

  return im_dec, la_dec

def _parse_and_decode_inference(filename, dataset_directory):
  im_dir = tf.cast(os.path.join(dataset_directory, 'images/'), tf.string)
  im_ext = tf.cast('.png', tf.string)

  im_filename = tf.string_join([im_dir, filename, im_ext])
  im_dec = tf.image.decode_jpeg(tf.read_file(im_filename))
  im_dec_raw = im_dec

  # Check if the image is in greyscale and convert to RGB if so
  greyscale_cond = tf.equal(tf.shape(im_dec)[-1], 1)
  im_dec = tf.cond(greyscale_cond,
                   lambda: tf.image.grayscale_to_rgb(im_dec),
                   lambda: tf.identity(im_dec))

  im_dec = tf.image.convert_image_dtype(im_dec, dtype=tf.float32)
  im_dec = from_0_1_to_m1_1(im_dec)

  return im_dec, im_filename, im_dec_raw

def _preprocess_images(image, label, params):
  if params.random_flip:
    label = tf.cast(tf.expand_dims(label, -1), tf.float32)
    im_la = tf.concat([image, label], axis=-1)
    im_la = tf.image.random_flip_left_right(im_la)

    image = im_la[..., 0:3]
    label = im_la[..., 3]

    label = tf.cast(label, tf.uint8)
    label = tf.squeeze(label)

    image.set_shape([params.height_input, params.width_input, 3])
    label.set_shape([params.height_input, params.width_input])

  return image, label

def _resize_images(image, label, height, width):
  """

  Args:
    image:
    label:
    height:
    width:

  Returns:

  """
  im_res = tf.image.resize_images(image, [height, width])
  la_res = tf.image.resize_images(label, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  la_res = tf.squeeze(la_res, axis=2)

  im_res.set_shape([height, width, 3])
  la_res.set_shape([height, width])

  return im_res, la_res

def _resize_images_inference(image, im_filename, im_raw, height, width):
  """

  Args:
    image:
    label:
    height:
    width:

  Returns:

  """
  im_res = tf.image.resize_images(image, [height, width])
  im_res.set_shape([height, width, 3])

  return im_res, im_filename, im_raw

def train_input(params):
  """

  Args:
    params:

  Returns:

  """
  dataset_directory = params.dataset_directory
  filelist_filepath = params.filelist_filepath
  filenames_string = tf.cast(filelist_filepath, tf.string)

  dataset = tf.data.TextLineDataset(filenames=filenames_string)

  dataset = dataset.map(
    functools.partial(_parse_and_decode, dataset_directory=dataset_directory),
    num_parallel_calls=30)

  dataset = dataset.map(
    functools.partial(_resize_images, height=params.height_input, width=params.width_input))

  dataset = dataset.map(
    functools.partial(_preprocess_images, params=params))

  # IMPORTANT: if Nb > 1, then shape of dataset elements must be the same
  dataset = dataset.batch(params.Nb)
  dataset = dataset.shuffle(100)
  dataset = dataset.repeat(None)

  return dataset

def evaluate_input(params):
  """

  Args:
    params:

  Returns:

  """
  dataset_directory = params.dataset_directory
  filelist_filepath = params.filelist_filepath
  filenames_string = tf.cast(filelist_filepath, tf.string)

  dataset = tf.data.TextLineDataset(filenames=filenames_string)

  dataset = dataset.map(
    functools.partial(_parse_and_decode, dataset_directory=dataset_directory),
    num_parallel_calls=30)

  dataset = dataset.map(
    functools.partial(_resize_images, height=params.height_input, width=params.width_input))

  # IMPORTANT: if Nb > 1, then shape of dataset elements must be the same
  dataset = dataset.batch(1)

  return dataset

def inference_input(params):
  """

  Args:
    params:

  Returns:

  """
  dataset_directory = params.dataset_directory
  filelist_filepath = params.filelist_filepath
  filenames_string = tf.cast(filelist_filepath, tf.string)

  dataset = tf.data.TextLineDataset(filenames=filenames_string)

  dataset = dataset.map(
    functools.partial(_parse_and_decode_inference, dataset_directory=dataset_directory),
    num_parallel_calls=30)

  dataset = dataset.map(
    functools.partial(_resize_images_inference, height=params.height_input, width=params.width_input))

  # IMPORTANT: if Nb > 1, then shape of dataset elements must be the same
  dataset = dataset.batch(1)

  return dataset