import os

from absl import app
from absl import flags
from sklearn.metrics import (accuracy_score, confusion_matrix,
    precision_recall_fscore_support, precision_recall_curve)
import tensorflow as tf

import dataloader
import eval_utils

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
    'num_classes', default=None,
    help=('Number of classes'))

flags.DEFINE_string(
    'exported_model_path', default=None,
    help=('Path to the directory containing exported files for the detection'
          ' model using the script exporter_main_v2.py')
)

flags.DEFINE_float(
    'detection_threshold', default=0.5,
    help=('Minimun confidence to consider detection as a point of interest')
)

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

flags.DEFINE_string(
    'results_file', default=None,
    help=('File name where the results will be stored.'))

flags.DEFINE_string(
    'model_name', default=None,
    help=('Name used to identify model on results fils. If no name is provided,'
          ' the checkpoint name will be used.'))

flags.mark_flag_as_required('validation_files')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('exported_model_path')

BATCH_SIZE = 1

def build_input_data():
  if FLAGS.validation_files.endswith('.csv'):
    if FLAGS.dataset_base_dir is None:
      raise RuntimeError('To use CSV files as input, you must specify'
                         ' --dataset_base_dir')

    input_data = dataloader.CSVInputProcessor(
      csv_file=FLAGS.validation_files,
      data_dir=FLAGS.dataset_base_dir,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=None,
      num_classes=FLAGS.num_classes,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.validation_files,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=None,
      num_classes=FLAGS.num_classes,
      num_instances=0
    )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def load_model():
  tf.keras.backend.clear_session()
  model = tf.saved_model.load(os.path.join(FLAGS.exported_model_path,
                                           'saved_model'))
  return model

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor).numpy()

def _decode_detection_prediction(confidence):
  if confidence >= FLAGS.detection_threshold:
    return 1
  else:
    return 0

def _detect_poi_agnostic(model, dataset):
  labels = []
  predictions = []
  detection_confidences = []
  count = 0

  for batch, label in dataset:
    detections = model(batch)
    detection_confidence = detections['detection_scores'][0][0].numpy()
    labels.append(_decode_one_hot(label[0]))
    predictions.append(_decode_detection_prediction(detection_confidence))
    detection_confidences.append(detection_confidence)

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return labels, predictions, detection_confidences

def eval_detector_as_binary_classifier(model, dataset):
  labels, predictions, probas_pred = _detect_poi_agnostic(model, dataset)

  accuracy = accuracy_score(labels, predictions)
  conf_matrix = confusion_matrix(labels, predictions)
  precision_recall_f1 = precision_recall_fscore_support(labels, predictions)
  prec_recall_curve = precision_recall_curve(labels, probas_pred)

  return accuracy, conf_matrix, precision_recall_f1, prec_recall_curve


def main(_):
  dataset = build_input_data()
  model = load_model()

  accuracy, conf_matrix, precision_recall_f1, prec_recall_curve = \
      eval_detector_as_binary_classifier(model, dataset)

  print(accuracy)
  print(conf_matrix)
  print(precision_recall_f1)
  print(prec_recall_curve)

  if FLAGS.results_file is not None:
    ckpt_name = FLAGS.exported_model_path.split('/')[-1]

    if FLAGS.model_name is not None:
      model_name = FLAGS.model_name
    else:
      model_name = ckpt_name
    eval_utils.save_results_to_file(FLAGS.results_file,
                                    model_name,
                                    ckpt_name,
                                    accuracy,
                                    conf_matrix,
                                    precision_recall_f1,
                                    prec_recall_curve)

if __name__ == '__main__':
  app.run(main)
