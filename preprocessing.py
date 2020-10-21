import tensorflow as tf

import randaugment

def flip(image):
  return tf.image.flip_left_right(image)

def normalize_image(image):
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

def preprocess_for_train(image,
                        output_size,
                        randaug_num_layers=None,
                        randaug_magnitude=None):
  image = tf.image.resize(image, size=(output_size, output_size))

  #random crop

  image = flip(image)

  if randaug_num_layers is not None and randaug_magnitude is not None:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = randaugment.distort_image_with_randaugment(image,
                                                       randaug_num_layers,
                                                       randaug_magnitude)

  image = normalize_image(image)

  return image

def preporocess_for_eval(image, output_size):

  # (center) crop
  #resize

  image = tf.image.resize(image, size=(output_size, output_size))

  image = normalize_image(image)

  return image

def preprocess_image(image,
                     output_size=224,
                     is_training=False,
                     randaug_num_layers=None,
                     randaug_magnitude=None):

  if is_training:
    return preprocess_for_train(image,
                                output_size,
                                randaug_num_layers,
                                randaug_magnitude)
  else:
    return preporocess_for_eval(image, output_size)
