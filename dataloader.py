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
              randaug_magnitude=None,
              use_fake_data=False,
              seed=None):
    self.csv_file = csv_file
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.is_training = is_training
    self.output_size = output_size
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude
    self.use_fake_data = use_fake_data
    self.seed = seed

  def make_source_dataset(self):
    csv_data = pd.read_csv(self.csv_file)
    num_instances = len(csv_data)
    num_classes = len(csv_data.category.unique())

    dataset = tf.data.Dataset.from_tensor_slices((
      csv_data.file_name,
      csv_data.category
    ))

    if self.is_training:
      dataset = dataset.shuffle(len(csv_data), seed=self.seed)
      dataset = dataset.repeat()

    def _load_image(file_name, label):
      image = tf.io.read_file(self.data_dir + file_name)
      image = tf.io.decode_jpeg(image, channels=3)
      label = tf.one_hot(label, num_classes)

      return image, label

    dataset = dataset.map(_load_image, num_parallel_calls=AUTOTUNE)

    def _preprocess_image(image, label):
      image = preprocessing.preprocess_image(image,
                                    output_size=self.output_size,
                                    is_training=self.is_training,
                                    randaug_num_layers=self.randaug_num_layers,
                                    randaug_magnitude=self.randaug_magnitude)

      return image, label

    dataset = dataset.map(_preprocess_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)

    if self.use_fake_data:
      dataset.take(1).repeat()

    return dataset, num_instances, num_classes
