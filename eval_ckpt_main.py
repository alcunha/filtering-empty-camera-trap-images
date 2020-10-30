import os

from absl import app
from absl import flags
import numpy as np

import dataloader
import model_builder
import train_image_classifier

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'validation_csv_file', default=None,
    help=('CSV file containing the list of images for evaluation'
          ' (CSV file must have two columns: filename and category id)'))

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_integer(
    'batch_size', default=1,
    help=('Batch size used during evaluation.'))

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes'))

def build_input_data(csv_file):
  input_data = dataloader.CSVInputProcessor(
    csv_file=csv_file,
    data_dir=FLAGS.dataset_base_dir,
    batch_size=FLAGS.batch_size,
    is_training=False,
    output_size=FLAGS.input_size,
    num_classes=FLAGS.num_classes,
  )

  return input_data.make_source_dataset()

def _generate_fake_instance():
  instance_shape = (FLAGS.input_size, FLAGS.input_size, 3)
  fake_instance = np.ones(instance_shape, dtype=np.float32)
  fake_labels = np.ones((FLAGS.num_classes,))

  return fake_instance, fake_labels

def _initialize_model_optimizer(model):
  fake_steps = 2
  fake_instance, fake_labels = _generate_fake_instance()
  x = np.array([fake_instance for i in range(fake_steps)])
  y = np.array([fake_labels for i in range(fake_steps)])

  model.fit(
    x, y, batch_size=FLAGS.batch_size, epochs=1, steps_per_epoch=fake_steps)

# dataset is required only to force load optimizer checkpoints
def load_model():
  model = model_builder.create(
    model_name=FLAGS.model_name,
    num_classes=FLAGS.num_classes,
    input_size=FLAGS.input_size
  )

  hparams = train_image_classifier.get_default_hparams()
  optimizer = train_image_classifier.generate_optimizer(hparams)
  loss_fn = train_image_classifier.generate_loss_fn(hparams)
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

  # workaround to fix 'Unresolved object in checkpoint' for optimizer variables
  _initialize_model_optimizer(model)

  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")

  model.load_weights(checkpoint_path)

  return model

def main(_):
  if FLAGS.validation_csv_file is None:
    raise RuntimeError('Must specify --validation_csv_file for evaluation.')

  if FLAGS.ckpt_dir is None:
    raise RuntimeError('Must specify --ckpt_dir for evaluation')

  if FLAGS.num_classes is None:
    raise RuntimeError('Must specify --num_classes for evaluation')

  if FLAGS.dataset_base_dir is None:
    raise RuntimeError('Must specify --dataset_base_dir for evaluation.')

  dataset, num_instances, _ = build_input_data(FLAGS.validation_csv_file)
  model = load_model()

if __name__ == '__main__':
  app.run(main)
