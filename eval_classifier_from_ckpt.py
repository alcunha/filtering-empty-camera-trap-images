import os

from absl import app
from absl import flags
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve
import numpy as np
import tensorflow as tf

import dataloader
import model_builder
import train_image_classifier

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'validation_files', default=None,
    help=('A file pattern for TFRecord files OR a CSV file containing the list'
          ' of images for evaluation (CSV file must have two columns: filename'
          ' and category id)'))

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory when using a CSV file'))

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

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

flags.mark_flag_as_required('validation_files')
flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')

def build_input_data():
  if FLAGS.validation_files.endswith('.csv'):
    if FLAGS.dataset_base_dir is None:
      raise RuntimeError('To use CSV files as input, you must specify'
                         ' --dataset_base_dir')

    input_data = dataloader.CSVInputProcessor(
      csv_file=FLAGS.validation_files,
      data_dir=FLAGS.dataset_base_dir,
      batch_size=FLAGS.batch_size,
      is_training=False,
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.validation_files,
      batch_size=FLAGS.batch_size,
      is_training=False,
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
      num_instances=0
    )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

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

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor).numpy()

def predict_binary_classifier(model, dataset):

  labels = []
  predictions = []
  detection_confidence = []
  count = 0

  for batch, label in dataset:
    prediction = model(batch, training=False)
    labels.append(_decode_one_hot(label[0]))
    predictions.append(_decode_one_hot(prediction[0]))
    detection_confidence.append(prediction[0].numpy()[1])

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return labels, predictions, detection_confidence

def eval_binary_classifier(model, dataset):
  labels, predictions, probas_pred = predict_binary_classifier(model, dataset)

  conf_matrix = confusion_matrix(labels, predictions)
  precision_recall = precision_recall_fscore_support(labels, predictions)
  prec_recall_curve = precision_recall_curve(labels, probas_pred)

  return conf_matrix, precision_recall, prec_recall_curve

def main(_):
  dataset = build_input_data()
  model = load_model()
  conf_matrix, precision_recall, prec_recall_curve = eval_binary_classifier(
                                                                model, dataset)

  print(conf_matrix)
  print(precision_recall)
  print(prec_recall_curve)

if __name__ == '__main__':
  app.run(main)
