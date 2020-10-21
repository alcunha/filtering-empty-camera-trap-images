from absl import app
from absl import flags

import dataloader
import utils

flags.DEFINE_string(
    'training_csv_file', default=None,
    help=('CSV file containing the list of images for training'
          ' (CSV file must have two columns: filename and category id)'))
flags.DEFINE_string(
    'validation_csv_file', default=None,
    help=('CSV file containing the list of images for evaluation'
          ' (CSV file must have two columns: filename and category id)'))

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory'))

flags.DEFINE_integer(
        'batch_size', default=32,
        help=('Batch size used during training.'))

flags.DEFINE_integer(
        'randaug_num_layers', default=None,
        help=('Number of operations to be applied by Randaugment'))

flags.DEFINE_integer(
        'randaug_magnitude', default=None,
        help=('Magnitude for operations on Randaugment.'))

FLAGS = flags.FLAGS


def build_input_data(csv_file, is_training=False):
  input_data = dataloader.CSVInputProcessor(
    csv_file=csv_file,
    data_dir=FLAGS.dataset_base_dir,
    batch_size=FLAGS.batch_size,
    is_training=is_training,
    randaug_num_layers=FLAGS.randaug_num_layers,
    randaug_magnitude=FLAGS.randaug_magnitude
  )

  return input_data.make_source_dataset()

def main(_):
  if FLAGS.training_csv_file is None:
    raise RuntimeError('Must specify --training_csv_file for train.')

  if FLAGS.dataset_base_dir is None:
    raise RuntimeError('Must specify --dataset_base_dir for train.')

  if utils.xor(FLAGS.randaug_num_layers is None,
               FLAGS.randaug_magnitude is None):
    raise RuntimeError('To apply Randaugment during training you must specify'
                       ' both --randaug_num_layers and --randaug_magnitude')

  dataset = build_input_data(FLAGS.training_csv_file,
                             is_training=True)

  for image, label in dataset.take(1):
    print(image.numpy().shape)
    print(label.numpy())

if __name__ == '__main__':
  app.run(main)
