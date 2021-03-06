# Copyright 2020 Fagner Cunha
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

r"""Tool to train classifiers.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import os
import random

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import dataloader
import model_builder
import train_image_classifier
import utils

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'training_files', default=None,
    help=('A file pattern for TFRecord files OR a CSV file containing the list'
          ' of images for training (CSV file must have two columns: filename'
          ' and category id)'))

flags.DEFINE_integer(
    'num_training_instances', default=None,
    help=('Number of training instances')
)

flags.DEFINE_string(
    'validation_files', default=None,
    help=('A file pattern for TFRecord files OR a CSV file containing the list'
          ' of images for evaluation (CSV file must have two columns: filename'
          ' and category id)'))

flags.DEFINE_integer(
    'num_validation_instances', default=None,
    help=('Number of validation instances')
)

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory when using a CSV file'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_bool(
    'use_label_smoothing', default=False,
    help=('Apply Label Smoothing to the labels during training')
)

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during training.'))

flags.DEFINE_integer(
    'randaug_num_layers', default=None,
    help=('Number of operations to be applied by Randaugment'))

flags.DEFINE_integer(
    'randaug_magnitude', default=None,
    help=('Magnitude for operations on Randaugment.'))

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_bool(
    'quant_aware_train', default=False,
    help=('Whether to use quantization aware training'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Path to weights checkpoint to be loaded into the model'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes to train the model on. If not passed, it will be'
           'inferred from data'))

flags.DEFINE_float(
    'lr', default=0.01,
    help=('Initial learning rate')
)

flags.DEFINE_float(
    'momentum', default=0,
    help=('Momentum for SGD optimizer')
)

flags.DEFINE_bool(
    'use_scaled_lr', default=True,
    help=('Scale the initial learning rate by batch size')
)

flags.DEFINE_bool(
    'use_cosine_decay', default=True,
    help=('Apply cosine decay during training')
)

flags.DEFINE_float(
    'warmup_epochs', default=0.3,
    help=('Duration of warmp of learning rate in epochs. It can be a'
          ' fractionary value as long will be converted to steps.')
)

flags.DEFINE_integer(
    'epochs', default=10,
    help=('Number of epochs to training for')
)

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments')
  )

flags.mark_flag_as_required('training_files')
flags.mark_flag_as_required('model_dir')

def build_csv_input_data(csv_file, is_training=False):
  if FLAGS.dataset_base_dir is None:
    raise RuntimeError('To use CSV files as input, you must specify'
                       ' --dataset_base_dir')

  input_data = dataloader.CSVInputProcessor(
    csv_file=csv_file,
    data_dir=FLAGS.dataset_base_dir,
    batch_size=FLAGS.batch_size,
    is_training=is_training,
    output_size=FLAGS.input_size,
    randaug_num_layers=FLAGS.randaug_num_layers,
    randaug_magnitude=FLAGS.randaug_magnitude,
    seed=FLAGS.random_seed,
  )

  return input_data.make_source_dataset()

def build_tfrecord_input_data(file_pattern, num_instances, is_training=False):
  if FLAGS.num_classes is None:
    raise RuntimeError('To use TFRecords as input, you must specify'
                       ' --num_classes')

  input_data = dataloader.TFRecordWBBoxInputProcessor(
    file_pattern=file_pattern,
    batch_size=FLAGS.batch_size,
    num_classes=FLAGS.num_classes,
    num_instances=num_instances,
    is_training=is_training,
    output_size=FLAGS.input_size,
    randaug_num_layers=FLAGS.randaug_num_layers,
    randaug_magnitude=FLAGS.randaug_magnitude,
    seed=FLAGS.random_seed,
  )

  return input_data.make_source_dataset()

def get_model(num_classes):
  model = model_builder.create(
    model_name=FLAGS.model_name,
    num_classes=num_classes,
    input_size=FLAGS.input_size,
    seed=FLAGS.random_seed
  )

  if FLAGS.ckpt_dir is not None: 
    checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
    model.load_weights(checkpoint_path)

  if FLAGS.quant_aware_train:
    model = tfmot.quantization.keras.quantize_model(model)

  return model

def train_model(model, train_data_and_size, val_data_and_size):

  if FLAGS.use_scaled_lr:
    lr = FLAGS.lr * FLAGS.batch_size / 256
  else:
    lr = FLAGS.lr

  _, train_size = train_data_and_size
  warmup_steps = int(FLAGS.warmup_epochs * (train_size // FLAGS.batch_size))

  hparams = train_image_classifier.get_default_hparams()
  hparams = hparams._replace(
    lr=lr,
    momentum=FLAGS.momentum,
    epochs=FLAGS.epochs,
    warmup_steps=warmup_steps,
    use_cosine_decay=FLAGS.use_cosine_decay,
    batch_size=FLAGS.batch_size,
    model_dir=FLAGS.model_dir,
    use_label_smoothing=FLAGS.use_label_smoothing
  )

  history = train_image_classifier.train_model(
    model,
    hparams,
    train_data_and_size,
    val_data_and_size
  )

  return history

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  if utils.xor(FLAGS.randaug_num_layers is None,
               FLAGS.randaug_magnitude is None):
    raise RuntimeError('To apply Randaugment during training you must specify'
                       ' both --randaug_num_layers and --randaug_magnitude')

  set_random_seeds()

  if FLAGS.training_files.endswith('.csv'):
    dataset, num_instances, num_classes = build_csv_input_data(
      FLAGS.training_files,
      is_training=True
    )
  else:
    if FLAGS.num_training_instances is None:
      raise RuntimeError('Must specify --num_training_instances when using'
                         'TFREcords for training')

    dataset, num_instances, num_classes = build_tfrecord_input_data(
      FLAGS.training_files,
      FLAGS.num_training_instances,
      is_training=True
    )

  if FLAGS.validation_files is not None:
    if FLAGS.validation_files.endswith('.csv'):
      val_dataset, val_num_instances, _ = build_csv_input_data(
        FLAGS.validation_files,
        is_training=False
      )
    else:
      if FLAGS.num_validation_instances is None:
        raise RuntimeError('Must specify --num_validation_instances when using'
                          'TFREcords for validation')

      val_dataset, val_num_instances, _ = build_tfrecord_input_data(
        FLAGS.validation_files,
        FLAGS.num_validation_instances,
        is_training=False
      )
  else:
    val_dataset = None
    val_num_instances = 0

  if FLAGS.num_classes is not None:
    num_classes = FLAGS.num_classes

  model = get_model(num_classes)

  model.summary()

  history = train_model(
    model,
    train_data_and_size=(dataset, num_instances),
    val_data_and_size=(val_dataset, val_num_instances)
  )

if __name__ == '__main__':
  app.run(main)
