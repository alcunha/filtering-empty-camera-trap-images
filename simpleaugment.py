import tensorflow as tf
import tensorflow_addons as tfa

import utils

def distort_color(image):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.random_brightness(image, max_delta=32. / 255.)
  image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
  image = tf.image.random_hue(image, max_delta=0.2)

  return tf.clip_by_value(image, 0.0, 1.0)

def random_rotation(image, deg=20):
  rotation_theta = utils.deg2rad(deg)

  random_deg = tf.random.uniform(
    shape=[1],
    minval=-rotation_theta,
    maxval=rotation_theta)

  image = tfa.image.rotate(image, random_deg, interpolation='BILINEAR')

  return image

def distort_image_with_simpleaugment(image):

  image = distort_color(image)
  image = random_rotation(image)

  return image
