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
"""

from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'exported_model_path', default=None,
    help=('Path to the directory containing exported files for the detection'
          ' model using the script exporter_main_v2.py')
)

flags.DEFINE_string(
    'output_model_file', default=None,
    help=('Name and path to tflite file model'))

flags.mark_flag_as_required('exported_model_path')
flags.mark_flag_as_required('output_model_file')

def convert_to_tflite():
  converter = tf.lite.TFLiteConverter.from_saved_model(
                  FLAGS.exported_model_path)
  tflite_model = converter.convert()

  with open(FLAGS.output_model_file, 'wb') as file_hdl:
    file_hdl.write(tflite_model)

def main(_):
  convert_to_tflite()
  print("Saving %s to %s" % (FLAGS.exported_model_path,
                             FLAGS.output_model_file))

if __name__ == '__main__':
  app.run(main)
