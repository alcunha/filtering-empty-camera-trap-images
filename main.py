from absl import app
from absl import flags

import dataloader
import model_builder
import train_image_classifier
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
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_bool(
    'use_label_smoothing', default=False,
    help=('Apply Label Smoothing to the labels during training')
)

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during training.'))

flags.DEFINE_integer(
    'randaug_num_layers', default=None,
    help=('Number of operations to be applied by Randaugment'))

flags.DEFINE_integer(
    'randaug_magnitude', default=None,
    help=('Magnitude for operations on Randaugment.'))

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_string(
    'model_dir', default='/tmp/ckp/',
    help=('Location of the model checkpoint files'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes to train the model on. If not passed, it will be'
           'inferred from data'))

flags.DEFINE_float(
    'lr', default=0.01,
    help=('Initial learning rate')
)

flags.DEFINE_float(
    'momentum', default=0,
    help=('Momentum for SGD optimizer')
)

flags.DEFINE_bool(
    'use_scaled_lr', default=True,
    help=('Scale the initial learning rate by batch size')
)

flags.DEFINE_bool(
    'use_cosine_decay', default=True,
    help=('Apply cosine decay during training')
)

flags.DEFINE_float(
    'warmup_epochs', default=1.5,
    help=('Duration of warmp of learning rate in epochs. It can be a'
          ' fractionary value as long will be converted to steps.')
)

flags.DEFINE_integer(
    'epochs', default=10,
    help=('Number of epochs to training for')
)

FLAGS = flags.FLAGS


def build_input_data(csv_file, is_training=False):
  input_data = dataloader.CSVInputProcessor(
    csv_file=csv_file,
    data_dir=FLAGS.dataset_base_dir,
    batch_size=FLAGS.batch_size,
    is_training=is_training,
    output_size=FLAGS.input_size,
    randaug_num_layers=FLAGS.randaug_num_layers,
    randaug_magnitude=FLAGS.randaug_magnitude
  )

  return input_data.make_source_dataset()

def get_model(num_classes):
  model = model_builder.create(
    model_name=FLAGS.model_name,
    num_classes=num_classes,
    input_size=FLAGS.input_size
  )

  return model

def train_model(model, train_data_and_size, val_data_and_size):

  if FLAGS.use_scaled_lr:
    lr = FLAGS.lr * FLAGS.batch_size / 256
  else:
    lr = FLAGS.lr

  _, train_size = train_data_and_size
  warmup_steps = int(FLAGS.warmup_epochs * (train_size // FLAGS.batch_size))

  hparams = train_image_classifier.get_default_hparams()
  hparams = hparams._replace(
    lr=lr,
    momentum=FLAGS.momentum,
    epochs=FLAGS.epochs,
    warmup_steps=warmup_steps,
    use_cosine_decay=FLAGS.use_cosine_decay,
    batch_size=FLAGS.batch_size,
    model_dir=FLAGS.model_dir,
    use_label_smoothing=FLAGS.use_label_smoothing
  )

  history = train_image_classifier.train_model(
    model,
    hparams,
    train_data_and_size,
    val_data_and_size
  )

  return history

def main(_):
  if FLAGS.training_csv_file is None:
    raise RuntimeError('Must specify --training_csv_file for train.')

  if FLAGS.dataset_base_dir is None:
    raise RuntimeError('Must specify --dataset_base_dir for train.')

  if FLAGS.model_dir is None:
    raise RuntimeError('Must specify --model_dir for train.')

  if utils.xor(FLAGS.randaug_num_layers is None,
               FLAGS.randaug_magnitude is None):
    raise RuntimeError('To apply Randaugment during training you must specify'
                       ' both --randaug_num_layers and --randaug_magnitude')

  dataset, num_instances, num_classes = build_input_data(
    FLAGS.training_csv_file,
    is_training=True
  )

  if FLAGS.validation_csv_file is not None:
    val_dataset, val_num_instances, _ = build_input_data(
      FLAGS.validation_csv_file,
      is_training=False
    )
  else:
    val_dataset = None
    val_num_instances = 0

  if FLAGS.num_classes is not None:
    num_classes = FLAGS.num_classes

  model = get_model(num_classes)

  model.summary()

  history = train_model(
    model,
    train_data_and_size=(dataset, num_instances),
    val_data_and_size=(val_dataset, val_num_instances)
  )

  print(history)

if __name__ == '__main__':
  app.run(main)
