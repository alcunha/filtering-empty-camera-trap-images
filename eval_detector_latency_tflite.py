import glob
import time

from absl import app
from absl import flags
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

try:
  import tensorflow.lite as tfl
except ImportError:
  import tflite_runtime.interpreter as tfl

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

def _resize_to_square_image(image, output_size):
  original_size = image.size
  ratio = float(output_size)/max(original_size)
  new_size = tuple([int(x*ratio) for x in original_size])
  im = image.resize(new_size, Image.ANTIALIAS)

  d_w = output_size - new_size[0]
  d_h = output_size - new_size[1]
  pad = (d_w//2, d_h//2, d_w - (d_w//2), d_h - (d_h//2))

  return ImageOps.expand(im, pad)

def load_image(image_path, height, width):
  image = Image.open(image_path).convert('RGB')
  image = _resize_to_square_image(image, max(height, width))

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

def _apply_detector(interpreter, image):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()

  bboxes = interpreter.get_tensor(output_details[0]['index'])
  confidences = interpreter.get_tensor(output_details[2]['index'])

  return bboxes, confidences

def _save_predictions_to_csv(file_names, predictions):
  preds = {
    'file_names': file_names,
    'predictions': predictions
  }
  df = pd.DataFrame.from_dict(preds, orient='index').transpose()
  df.to_csv(FLAGS.predictions_csv_file, index=False)

def eval_model():
  image_count = 0
  total_elapsed_time = 0
  predictions = []
  bboxes_list = []
  interpreter, height, width = load_model_interpreter(FLAGS.model)
  image_list = glob.glob(FLAGS.images_patern)

  for image_path in image_list:
    image = load_image(image_path, height, width)

    start_time = time.time()
    bboxes, confidences = _apply_detector(interpreter, image)
    elapsed_ms = (time.time() - start_time) * 1000

    total_elapsed_time += elapsed_ms
    image_count += 1
    predictions.append(confidences)
    bboxes_list.append(bboxes)

  if FLAGS.predictions_csv_file is not None:
    _save_predictions_to_csv(image_list, predictions)

  return total_elapsed_time/image_count

def main(_):
  avg_elapsed_time = eval_model()
  print("Avareged elapsed time: %fms" % (avg_elapsed_time))

if __name__ == '__main__':
  app.run(main)
