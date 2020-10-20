import tensorflow as tf

def preprocess_image(image, output_size=224):
  image = tf.image.resize(image, size=(output_size, output_size))

  return image