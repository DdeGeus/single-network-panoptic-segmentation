import tensorflow as tf
from tensorflow.python.client import timeline
import os
import time
import argparse
from model import PanopticSegmentationModel
from input_pipelines.input_pipeline_cityscapes import train_input
from input_pipelines.input_pipeline_cityscapes_panoptic_new import train_input as train_input_panoptic
from utils.load_json_to_params import load_json_to_params
from utils.utils import replace_initializers
from config import Params

def get_arguments():
  parser = argparse.ArgumentParser(description="Panoptic-Slim Network")
  parser.add_argument("--json_path", type=str, default='',
                      help="The path to the json file containing the parameters")

  return parser.parse_args()

def train(params):
  start_step = 0
  coord = tf.train.Coordinator()

  if params.apply_instance_branch:
    dataset = train_input_panoptic(params)
    iterator = dataset.make_one_shot_iterator()

    images, labels, instance_gt = iterator.get_next()

    model = PanopticSegmentationModel(images, labels, params, instance_gt)

  else:
    dataset = train_input(params)
    iterator = dataset.make_one_shot_iterator()

    images, labels = iterator.get_next()

    model = PanopticSegmentationModel(images, labels, params)

  prediction_dict = model.predict()
  loss = model.loss(prediction_dict, save_image_summaries=True)

  global_step_tensor = tf.train.create_global_step()
  step_current = tf.placeholder(dtype=tf.float32, shape=())
  base_lr = params.learning_rate

  if params.lr_schedule == 'constant':
    learning_rate = base_lr
  elif params.lr_schedule == 'poly':
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_current / params.num_steps), params.lr_power))
  elif params.lr_schedule == 'piecewise_constant' or params.lr_schedule == 'stepwise':
    lr_factor = params.lr_step_factor
    learning_rate = tf.train.piecewise_constant(
      step_current,
      params.lr_boundaries,
      [base_lr, base_lr*lr_factor, base_lr*(lr_factor**2)])
  else:
    learning_rate = base_lr

  vars_trainable = [v for v in tf.trainable_variables() if
                    ('beta' not in v.name and 'gamma' not in v.name) or params.train_beta_gamma]

  tf.summary.scalar('learning_rate', learning_rate, family='optimizer')

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  with tf.control_dependencies(update_ops):
    opt_main = tf.train.MomentumOptimizer(learning_rate, params.momentum)
    grads = tf.gradients(loss, vars_trainable)
    train_op = opt_main.apply_gradients(zip(grads, vars_trainable), global_step=global_step_tensor)

  config = tf.ConfigProto()
  sess = tf.Session(config=config)

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(params.log_dir + '/train',
                                       sess.graph)

  save_var = tf.global_variables()
  saver = tf.train.Saver(var_list=save_var, max_to_keep=10)

  if params.resnet_init:
    replace_initializers(params)

  # Run the initializer
  init = tf.global_variables_initializer()
  local_init = tf.local_variables_initializer()
  sess.run(init)
  sess.run(local_init)

  # Log metadata
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                          output_partition_graphs=True)
  run_metadata = tf.RunMetadata()

  START_FROM_CKPT = False
  if START_FROM_CKPT:
    start_step = sess.run(tf.train.get_global_step()) - 1
    print("Start step:", start_step)

  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Iterate over training steps.
  for step in range(start_step, params.num_steps):
    start_time = time.time()

    feed_dict = {step_current: step}
    if step % params.save_summaries == 0:
      loss_value, summary, _ = sess.run([loss, merged, train_op], feed_dict=feed_dict)
      train_writer.add_summary(summary, step)
      train_writer.flush()
    elif step == 1:
      loss_value, _ = sess.run([loss, train_op],
                               options=options,
                               feed_dict=feed_dict,
                               run_metadata=run_metadata)
      # Create the Timeline object, and write it to a json file
      fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      chrome_trace = fetched_timeline.generate_chrome_trace_format()
      with open(os.path.join(params.log_dir, 'timeline_01.json'), 'w') as f:
        f.write(chrome_trace)
      with open(os.path.join(params.log_dir, 'mem_info.json'), 'w') as f:
        f.write(str(run_metadata))

    else:
      loss_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)

    if step % params.ckpt_save_steps == 0:
      model.save(saver, sess, params.log_dir, step)

    duration = time.time() - start_time
    print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
  args = get_arguments()
  params = Params()
  params = load_json_to_params(params, args.json_path)
  params.dataset_directory = '/home/ddegeus/datasets/Cityscapes/training/'
  params.filelist_filepath = '/home/ddegeus/datasets/Cityscapes/training/panoptic/filenames.lst'

  params.is_training = True
  params.batch_norm_istraining = True
  print(params)
  train(params)