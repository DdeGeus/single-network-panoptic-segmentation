# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Tensorflow Object Detection API code adapted by Daan de Geus

import tensorflow as tf

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  with tf.variable_scope("IndicesToDenseVector"):
    size = tf.to_int32(size)
    zeros = tf.ones([size], dtype=dtype) * default_value
    values = tf.ones_like(indices, dtype=dtype) * indices_value

    return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                             [zeros, values])

def replace_initializers(params, scope='replaced_initializers'):
  # currently supported initialization:
  #   0) start training from scratch
  #   1) initialize from init_ckpt_path (log_dir has to be empty of checkpoints)
  #   2) continue training from log_dir

  with tf.name_scope(scope), tf.device('/cpu:0'):
    ## initialize from checkpoint, e.g. trained on ImageNet
    # an empty string '' is False
    if params.init_ckpt_path:
      # the best we can do is to initialize from the checkpoint as much variables as possible
      # so we find the mapping from checkpoint names to model names
      # assumes names in model are extended with a prefix from names in checkpoint
      # e.g.
      # in checkpoint: resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
      # in model: feature_extractor/base/resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights

      # list of (name, shape) of checkpoint variables
      ckpt_vars = tf.train.list_variables(params.init_ckpt_path)
      # list of tf.Variable of model variables
      global_vars = tf.global_variables()

      # checkpoint variable name --> model variable mappings

      exclude = ['train_ops', 'ExponentialMovingAverage',
                  'Momentum', 'classifier', 'extension']
      if not params.psp_module:
        exclude.append('psp')
      # if not params.aspp_module:
      #   exclude.append('aspp')
      var_dict = dict()
      extra_var_dict = dict()
      # print("GLOBAL:", global_vars)
      for gv in global_vars:
        for cvn, cvs in ckpt_vars:
          for exc in exclude:
            if exc in gv.name:
              break
          else:
            if cvn in gv.name and tf.TensorShape(cvs).is_compatible_with(gv.shape):
              if cvn not in var_dict.keys():
                var_dict[cvn] = gv
              else:
                extra_var_dict[cvn] = gv
      #
      # print("VAR DICT:")
      # print(var_dict)
      # print("EXTRA VAR DICT:")
      # print(extra_var_dict)

      # extra_vars_to_init = set(global_vars).difference(set(var_dict.values()))

      # for now init_from_checkpoint doesn't support DistributedValues (TF bug, error)
      # so do a scan and unwrap DistibutedValues
      # # for k, v in var_dict.items():
      # #   if isinstance(v, DistributedValues):
      # #     var_dict[k] = v.get()
      # #   else:
      # #     # keep default behavior
      # #     pass
      # suppress INFO logging messages
      # with _temp_verbosity(tf.logging.WARN):
      tf.train.init_from_checkpoint(params.init_ckpt_path, var_dict)
      tf.train.init_from_checkpoint(params.init_ckpt_path, extra_var_dict)
    else:
      # start from scratch or continue from log_dir (managed by estimator)
      pass
