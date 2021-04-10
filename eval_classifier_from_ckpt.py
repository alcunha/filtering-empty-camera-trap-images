# Copyright 2020 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Tool to evaluate classifiers.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import os
import random

from absl import app
from absl import flags
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
    precision_recall_fscore_support, precision_recall_curve)
import pandas as pd
import tensorflow as tf

import dataloader
import eval_utils

os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

flags.DEFINE_string(
    'results_file', default=None,
    help=('File name where the results will be stored.'))

flags.DEFINE_string(
    'predictions_csv_file', default=None,
    help=('File name to save model predictions.'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('validation_files')
flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')

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
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
      provide_filename=True,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.validation_files,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=FLAGS.input_size,
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

  if not os.path.exists(os.path.dirname(FLAGS.predictions_csv_file)):
    os.makedirs(os.path.dirname(FLAGS.predictions_csv_file))

  df = pd.DataFrame.from_dict(preds, orient='index').transpose()
  df.to_csv(FLAGS.predictions_csv_file, index=False)

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor).numpy()

def predict_binary_classifier(model, dataset):
  file_names = []
  labels = []
  predictions = []
  detection_confidences = []
  count = 0

  for batch, metadata in dataset:
    prediction = model(batch, training=False)
    label, file_name = metadata
    labels.append(_decode_one_hot(label[0]))
    file_names.append(file_name[0].numpy())
    predictions.append(_decode_one_hot(prediction[0]))
    detection_confidences.append(prediction[0].numpy()[1])

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  if FLAGS.predictions_csv_file is not None:
    _save_predictions_to_csv(file_names, labels, detection_confidences)

  return labels, predictions, detection_confidences

def eval_binary_classifier(model, dataset):
  labels, predictions, probas_pred = predict_binary_classifier(model, dataset)

  accuracy = accuracy_score(labels, predictions)
  conf_matrix = confusion_matrix(labels, predictions)
  precision_recall_f1 = precision_recall_fscore_support(labels, predictions)
  prec_recall_curve = precision_recall_curve(labels, probas_pred)

  return accuracy, conf_matrix, precision_recall_f1, prec_recall_curve

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()
  dataset = build_input_data()
  model = eval_utils.load_model_from_checkpoint(FLAGS.model_name,
                                                FLAGS.num_classes,
                                                FLAGS.input_size,
                                                FLAGS.ckpt_dir)
  accuracy, conf_matrix, precision_recall_f1, prec_recall_curve = \
      eval_binary_classifier(model, dataset)

  print(accuracy)
  print(conf_matrix)
  print(precision_recall_f1)
  print(prec_recall_curve)

  if FLAGS.results_file is not None:
    ckpt_name = FLAGS.ckpt_dir.split('/')[-1]
    eval_utils.save_results_to_file(FLAGS.results_file,
                                    FLAGS.model_name,
                                    ckpt_name,
                                    accuracy,
                                    conf_matrix,
                                    precision_recall_f1,
                                    prec_recall_curve)

if __name__ == '__main__':
  app.run(main)
