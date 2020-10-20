import tensorflow as tf
import pandas as pd

import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE

class CSVInputProcessor:

  def __init__(self,
              csv_file,
              data_dir,
              batch_size,
              is_training=False,
              output_size=224,
              randaug_num_layers=None,
              randaug_magnitude=None):
    self.csv_file = csv_file
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.is_training = is_training
    self.output_size = output_size
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude

  def make_source_dataset(self):
    csv_data = pd.read_csv(self.csv_file)
    dataset = tf.data.Dataset.from_tensor_slices((
      csv_data.file_name,
      csv_data.category
    ))

    if self.is_training:
      dataset = dataset.shuffle(len(csv_data))
      dataset = dataset.repeat()

    def _load_image(file_name, label):
      image = tf.io.read_file(self.data_dir + file_name)
      image = tf.image.decode_jpeg(image, channels=3)

      return image, label
    
    dataset = dataset.map(_load_image, num_parallel_calls=AUTOTUNE)

    def _preprocess_image(image, label):
      image = preprocessing.preprocess_image(image)

      return image, label

    dataset = dataset.map(_preprocess_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)

    return dataset
