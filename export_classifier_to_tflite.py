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

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('output_model_file')

def convert_to_tflite():
  model = eval_utils.load_model_from_checkpoint(FLAGS.model_name,
                                                FLAGS.num_classes,
                                                FLAGS.input_size,
                                                FLAGS.ckpt_dir)

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
