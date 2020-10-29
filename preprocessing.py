from absl import flags

import tensorflow as tf

import randaugment
import simpleaugment

flags.DEFINE_bool(
    'normalize_input', default=True,
    help=('Normalize inputs using ImageNet mean and std'))

flags.DEFINE_bool(
    'use_simple_augment', default=False,
    help=('Use simple data augmentation with random hue, saturation, '
          ' brightness, and rotation. If Randaugment parameters is used, this'
          ' options is ignored.'))

FLAGS = flags.FLAGS

def random_crop(image,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.65, 1],
                min_object_covered=0.5,
                max_attempts=100,
                seed=0):

  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      area_range=area_range,
      aspect_ratio_range=aspect_ratio_range,
      use_image_if_no_bounding_boxes=True,
      max_attempts=max_attempts,
      seed=seed
  )

  offset_height, offset_width, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.crop_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
  )

  return image

def flip(image, seed=None):
  return tf.image.random_flip_left_right(image, seed)

def normalize_image(image):
  tf.compat.v1.logging.info('Normalizing inputs.')
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  mean = tf.constant([0.485, 0.456, 0.406])
  mean = tf.expand_dims(mean, axis=0)
  mean = tf.expand_dims(mean, axis=0)
  image = image - mean

  std = tf.constant([0.229, 0.224, 0.225])
  std = tf.expand_dims(std, axis=0)
  std = tf.expand_dims(std, axis=0)
  image = image/std

  return image

def resize_image(image, output_size):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, size=(output_size, output_size))

  return image

def preprocess_for_train(image,
                        output_size,
                        randaug_num_layers=None,
                        randaug_magnitude=None,
                        seed=None):

  image = random_crop(image, seed)
  image = resize_image(image, output_size)
  image = flip(image, seed)

  if randaug_num_layers is not None and randaug_magnitude is not None:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = randaugment.distort_image_with_randaugment(image,
                                                       randaug_num_layers,
                                                       randaug_magnitude)
  else:
    if FLAGS.use_simple_augment:
      image = simpleaugment.distort_image_with_simpleaugment(image, seed)

  if FLAGS.normalize_input:
    image = normalize_image(image)
  else:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  return image

def preprocess_for_eval(image, output_size):

  image = resize_image(image, output_size)

  if FLAGS.normalize_input:
    image = normalize_image(image)
  else:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  return image

def preprocess_image(image,
                     output_size=224,
                     is_training=False,
                     randaug_num_layers=None,
                     randaug_magnitude=None,
                     seed=None):

  if is_training:
    return preprocess_for_train(image,
                                output_size,
                                randaug_num_layers,
                                randaug_magnitude,
                                seed)
  else:
    return preprocess_for_eval(image, output_size)
