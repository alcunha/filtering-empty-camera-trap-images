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

from absl import app
from absl import flags
import tensorflow as tf

import dataloader
import eval_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_string(
    'output_model_file', default=None,
    help=('Name and path to tflite file model'))

flags.DEFINE_bool(
    'use_quantization', default=False,
    help=('Apply pos-training quantization during exporting process'))

flags.DEFINE_bool(
    'quant_aware_train', default=False,
    help=('Whether the model was trained with quantization aware training'))

flags.DEFINE_string(
    'representative_dataset', default=None,
    help=('A file pattern for TFRecord files OR a CSV file containing the list'
          ' of images used to calibrate the range of floating-point tensors'
          ' when using quantization (CSV file must have two columns: filename'
          ' and category id)'))

flags.DEFINE_integer(
    'num_samples', default=500,
    help=('Number of samples used from representative dataset.'))

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory when using a CSV file'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('output_model_file')

BATCH_SIZE = 1

def _build_dataset():
  if FLAGS.representative_dataset.endswith('.csv'):
    if FLAGS.dataset_base_dir is None:
      raise RuntimeError('To use CSV files as input, you must specify'
                         ' --dataset_base_dir')

    input_data = dataloader.CSVInputProcessor(
      csv_file=FLAGS.representative_dataset,
      data_dir=FLAGS.dataset_base_dir,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.representative_dataset,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
      num_instances=0,
    )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def convert_to_tflite():
  model = eval_utils.load_model_from_checkpoint(FLAGS.model_name,
                                                FLAGS.num_classes,
                                                FLAGS.input_size,
                                                FLAGS.ckpt_dir,
                                                FLAGS.quant_aware_train)

  converter = tf.lite.TFLiteConverter.from_keras_model(model)

  if FLAGS.use_quantization:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if FLAGS.representative_dataset is not None:
      dataset = _build_dataset()

      def representative_dataset():
        for image, _ in dataset.take(FLAGS.num_samples):
          yield [image]

      converter.representative_dataset = representative_dataset

  tflite_model = converter.convert()

  with open(FLAGS.output_model_file, 'wb') as file_hdl:
    file_hdl.write(tflite_model)

def main(_):
  convert_to_tflite()
  print("Saving %s (%d) to %s" % (FLAGS.model_name,
                                  FLAGS.input_size,
                                  FLAGS.output_model_file))

if __name__ == '__main__':
  app.run(main)
