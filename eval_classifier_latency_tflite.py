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

r"""Tool to evaluate classifiers latency on edge devices.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import os
import glob
import time

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import pandas as pd

try:
  import tensorflow.lite as tfl
except ImportError:
  import tflite_runtime.interpreter as tfl

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', default=None,
    help=('File path of .tflite file.')
)

flags.DEFINE_string(
    'images_patern', default=None,
    help=('A file pattern for images to be used during evaluation.')
)

flags.DEFINE_enum(
    'input_scale_mode', default='float32',
    enum_values=['tf_mode', 'torch_mode', 'uint8', 'float32'],
    help=('Mode for scaling input: tf_mode scales image between -1 and 1;'
          ' torch_mode normalizes inputs using ImageNet mean and std using'
          ' float32 input format; uint8 uses image on scale 0-255; float32'
          ' uses image on scale 0-1'))

flags.DEFINE_string(
    'predictions_csv_file', default=None,
    help=('File name to save model predictions.')
)

flags.mark_flag_as_required('model')
flags.mark_flag_as_required('images_patern')

def normalize_image(image):
  image = np.asarray(image, dtype=np.float32)
  image = image/255

  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  mean = np.expand_dims(mean, axis=0)
  mean = np.expand_dims(mean, axis=0)
  image = image - mean

  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  std = np.expand_dims(std, axis=0)
  std = np.expand_dims(std, axis=0)
  image = image/std

  return image

def scale_input_tf_mode(image):
  image = np.asarray(image, dtype=np.float32)
  image /= 127.5
  image -= 1.

  return image

def load_image(image_path, height, width):
  image = Image.open(image_path).convert('RGB').resize((width, height),
                                                        Image.ANTIALIAS)

  if FLAGS.input_scale_mode == 'torch_mode':
    image = normalize_image(image)
  elif FLAGS.input_scale_mode == 'tf_mode':
    image = scale_input_tf_mode(image)
  elif FLAGS.input_scale_mode == 'uint8':
    image = np.asarray(image, dtype=np.uint8)
  else:
    image = np.asarray(image, dtype=np.float32)
    image = image/255

  image = np.expand_dims(image, axis=0)

  return image

def load_model_interpreter(model_path):
  interpreter = tfl.Interpreter(model_path)
  interpreter.allocate_tensors()

  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  return interpreter, input_height, input_width

def classify_image(interpreter, image):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  output_data = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

  return output_data

def _save_predictions_to_csv(file_names, predictions):
  preds = {
    'file_names': file_names,
    'predictions': predictions
  }

  if not os.path.exists(os.path.dirname(FLAGS.predictions_csv_file)):
    os.makedirs(os.path.dirname(FLAGS.predictions_csv_file))

  df = pd.DataFrame.from_dict(preds, orient='index').transpose()
  df.to_csv(FLAGS.predictions_csv_file, index=False)

def eval_model():
  image_count = 0
  total_elapsed_time = 0
  predictions = []
  interpreter, height, width = load_model_interpreter(FLAGS.model)
  image_list = glob.glob(FLAGS.images_patern)

  for image_path in image_list:
    image = load_image(image_path, height, width)

    start_time = time.time()
    preds = classify_image(interpreter, image)
    elapsed_ms = (time.time() - start_time) * 1000

    total_elapsed_time += elapsed_ms
    image_count += 1
    predictions.append(preds)

  if FLAGS.predictions_csv_file is not None:
    _save_predictions_to_csv(image_list, predictions)

  return total_elapsed_time/image_count

def main(_):
  avg_elapsed_time = eval_model()
  print("Avareged elapsed time: %fms" % (avg_elapsed_time))

if __name__ == '__main__':
  app.run(main)
