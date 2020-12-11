from absl import app
from absl import flags
from sklearn.metrics import (accuracy_score, confusion_matrix,
    precision_recall_fscore_support, precision_recall_curve)
import pandas as pd
import tensorflow as tf

import dataloader
import eval_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', default=None,
    help=('File path of .tflite file.')
)

flags.DEFINE_string(
    'model_name', default=None,
    help=('Model name of the archtecture. Used only on results file.'))

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

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

flags.DEFINE_string(
    'results_file', default=None,
    help=('File name where the results will be stored.'))

flags.DEFINE_string(
    'predictions_csv_file', default=None,
    help=('File name to save model predictions.')
)

flags.mark_flag_as_required('validation_files')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('num_classes')

BATCH_SIZE = 1

def build_input_data(input_size):
  if FLAGS.validation_files.endswith('.csv'):
    if FLAGS.dataset_base_dir is None:
      raise RuntimeError('To use CSV files as input, you must specify'
                         ' --dataset_base_dir')

    input_data = dataloader.CSVInputProcessor(
      csv_file=FLAGS.validation_files,
      data_dir=FLAGS.dataset_base_dir,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=input_size,
      num_classes=FLAGS.num_classes,
      provide_filename=True,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.validation_files,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=input_size,
      num_classes=FLAGS.num_classes,
      num_instances=0,
      provide_filename=True,
    )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def _save_predictions_to_csv(file_names, labels, predictions):
  preds = {
    'file_names': file_names,
    'labels': labels,
    'predictions': predictions
  }

  df = pd.DataFrame.from_dict(preds, orient='index').transpose()
  df.to_csv(FLAGS.predictions_csv_file, index=False)

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor).numpy()

def classify_image(interpreter, image):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()

  return interpreter.get_tensor(output_details[0]['index'])

def predict_binary_classifier(interpreter, dataset):
  file_names = []
  labels = []
  predictions = []
  detection_confidence = []
  count = 0

  for batch, metadata in dataset:
    prediction = classify_image(interpreter, batch)
    label, file_name = metadata
    labels.append(_decode_one_hot(label[0]))
    file_names.append(file_name[0].numpy())
    predictions.append(_decode_one_hot(prediction[0]))
    detection_confidence.append(prediction[0][1])

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  if FLAGS.predictions_csv_file is not None:
    _save_predictions_to_csv(file_names, labels, detection_confidence)

  return labels, predictions, detection_confidence

def eval_binary_classifier(interpreter, dataset):
  labels, predictions, probas_pred = predict_binary_classifier(interpreter,
                                                               dataset)

  accuracy = accuracy_score(labels, predictions)
  conf_matrix = confusion_matrix(labels, predictions)
  precision_recall_f1 = precision_recall_fscore_support(labels, predictions)
  prec_recall_curve = precision_recall_curve(labels, probas_pred)

  return accuracy, conf_matrix, precision_recall_f1, prec_recall_curve

def load_model_interpreter(model_path):
  interpreter = tf.lite.Interpreter(model_path)
  interpreter.allocate_tensors()

  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  return interpreter, input_height, input_width

def main(_):
  interpreter, input_size, _ = load_model_interpreter(FLAGS.model)
  dataset = build_input_data(input_size)

  accuracy, conf_matrix, precision_recall_f1, prec_recall_curve = \
      eval_binary_classifier(interpreter, dataset)

  print(accuracy)
  print(conf_matrix)
  print(precision_recall_f1)
  print(prec_recall_curve)

  if FLAGS.results_file is not None:
    ckpt_name = FLAGS.model.split('/')[-1]
    ckpt_name = ckpt_name.split('.')[0]

    if FLAGS.model_name is not None:
      model_name = FLAGS.model_name
    else:
      model_name = FLAGS.model

    eval_utils.save_results_to_file(FLAGS.results_file,
                                    model_name,
                                    ckpt_name,
                                    accuracy,
                                    conf_matrix,
                                    precision_recall_f1,
                                    prec_recall_curve)

if __name__ == '__main__':
  app.run(main)
