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
r""" Tool to export detector from a savedModel to TFLite

If you get an error during the process, install the package tf-nightly

For models from TensorFlow Object Detection API, it's recommended to export the
SavedModel using the script export_tflite_graph_tf2.py which provides some
options to optimize model such as limiting the number of detections.
"""

from absl import app
from absl import flags
import tensorflow as tf

import dataloader

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'exported_model_path', default=None,
    help=('Path to the directory containing exported files for the detection'
          ' model using the script exporter_main_v2.py including "saved_model"'
          ' folder.')
)

flags.DEFINE_string(
    'output_model_file', default=None,
    help=('Name and path to tflite file model'))

flags.DEFINE_bool(
    'use_quantization', default=False,
    help=('Apply pos-training quantization during exporting process'))

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

flags.DEFINE_integer(
    'input_size', default=None,
    help=('Input size of the model'))

flags.DEFINE_integer(
    'num_classes', default=2,
    help=('Number of classes for dataset'))

flags.DEFINE_bool(
    'resize_with_pad', default=False,
    help=('Apply padding when resizing image'))

flags.mark_flag_as_required('exported_model_path')
flags.mark_flag_as_required('output_model_file')

BATCH_SIZE = 1

def _build_dataset():
  if FLAGS.input_size is None:
    raise RuntimeError('To use a representative dataset, you must specify'
                         ' --input_size')

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
      resize_with_pad=FLAGS.resize_with_pad,
      num_classes=FLAGS.num_classes,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.representative_dataset,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=FLAGS.input_size,
      resize_with_pad=FLAGS.resize_with_pad,
      num_classes=FLAGS.num_classes,
      num_instances=0,
    )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def convert_to_tflite():
  converter = tf.lite.TFLiteConverter.from_saved_model(
                  FLAGS.exported_model_path)

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
  print("Saving %s to %s" % (FLAGS.exported_model_path,
                             FLAGS.output_model_file))

if __name__ == '__main__':
  app.run(main)
