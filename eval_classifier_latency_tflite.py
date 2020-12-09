import glob
import time

from absl import app
from absl import flags
import numpy as np
from PIL import Image

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
    enum_values=['tf_mode', 'torch_mode', 'unit8', 'float32'],
    help=('Mode for scaling input: tf_mode scales image between -1 and 1;'
          ' torch_mode normalizes inputs using ImageNet mean and std using'
          ' float32 input format; unit8 uses image on scale 0-255; float32'
          ' uses image on scale 0-1'))

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
  elif FLAGS.input_scale_mode == 'unit8':
    image = np.asarray(image, dtype=np.unit8)
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
    predictions.append({
      'image': image_path,
      'predictions': preds
    })

  return total_elapsed_time/image_count, predictions

def main(_):
  avg_elapsed_time, _ = eval_model()
  print("Avareged elapsed time: %fms" % (avg_elapsed_time))

if __name__ == '__main__':
  app.run(main)
