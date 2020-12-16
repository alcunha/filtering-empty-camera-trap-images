r""" Tool to export detector from a savedModel to TFLite

If you get an error during the process, install the package tf-nightly
"""

import os

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
  saved_model_dir = os.path.join(FLAGS.exported_model_path, 'saved_model')
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  with open(FLAGS.output_model_file, 'wb') as file_hdl:
    file_hdl.write(tflite_model)

def main(_):
  convert_to_tflite()
  model_name = FLAGS.exported_model_path.split('/')[-1]
  print("Saving %s to %s" % (model_name, FLAGS.output_model_file))

if __name__ == '__main__':
  app.run(main)
