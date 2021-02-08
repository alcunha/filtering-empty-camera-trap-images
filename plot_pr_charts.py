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

import os

import collections

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

import eval_utils
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'results_patern', default=None,
    help=('File patern for experiments results files'))

flags.DEFINE_string(
    'charts_path', default=None,
    help=('Path where the charts will be saved on')
)

flags.DEFINE_list(
    'chart_format_list', default=['pdf'],
    help=('List of file formats to save charts on')
)

flags.DEFINE_bool(
    'log_pr_auc', default=True,
    help=('Log Precision-Recall AUC for each model')
)

flags.DEFINE_multi_float(
    'log_prec_at_pr', default=None,
    help=('Log Precision at Recall levels')
)

flags.DEFINE_bool(
    'show_random_guess', default=False,
    help=('Show line for random guess')
)

flags.DEFINE_multi_float(
    'threshold_dots', default=None,
    help=('Include dots for thrsholds on chart')
)

flags.mark_flag_as_required('results_patern')
flags.mark_flag_as_required('charts_path')

ClassifierResults = collections.namedtuple("ClassifierResults", [
    'model_name', 'ckpt_name', 'accuracy', 'confusion_matrix',
    'precision_recall_f1_per_class', 'precision_recall_curve'
  ])

model_names = {'efficientdet-d0': 'EfficientDet-D0',
               'efficientnet-b0': 'EfficientNet-B0',
               'efficientnet-b3': 'EfficientNet-B3',
               'mobilenetv2': 'MobileNetV2',
               'mobilenetv2_a0.50': 'MobileNetV2-0.50',
               'mobilenetv2_a0.75': 'MobileNetV2-0.75',
               'ssdlite-mobilenetv2': 'SSDLite+MobileNetV2'}

default_model_resolutions = {'efficientdet-d0': 512,
                             'efficientnet-b0': 224,
                             'efficientnet-b3': 300,
                             'mobilenetv2': 224,
                             'ssdlite-mobilenetv2': 320}

default_models_sorter = ['EfficientNet-B0 (224)',
                         'EfficientNet-B3 (300)',
                         'MobileNetV2 (224)',
                         'MobileNetV2-0.50 (320)',
                         'MobileNetV2-0.75 (320)',
                         'MobileNetV2 (320)',
                         'SSDLite+MobileNetV2 (320)',
                         'EfficientDet-D0 (512)']

def _format_model_name(model, resolution=None):
  if model in model_names.keys():
    formatted_name = model_names[model]
  else:
    formatted_name = model

  if resolution is not None:
    return '%s (%s)' % (formatted_name, resolution)
  else:
    return formatted_name

def _get_resolution_from_ckp_name(model, ckp_name):
  try:
    resolution = ckp_name.split('_')[2]
    resolution = int(resolution)
  except ValueError:
    resolution = default_model_resolutions[model]

  return resolution

def _get_test_set_id_from_results_filename(filename):
  test_set_id = filename.split('/')[-1]
  test_set_id = test_set_id.split('.')[0]

  return test_set_id

def _calculate_no_skill_model(df, test_set_id):
  df = df[df.test_set_id == test_set_id]
  qty_class0 = df.precision_recall_f1_per_class.iloc[0][3][0]
  qty_class1 = df.precision_recall_f1_per_class.iloc[0][3][1]

  return qty_class1 / (qty_class0 + qty_class1)

def _sort_models_by_list(df, models_sorter):
  df = df.copy()
  df.fancy_model_name = df.fancy_model_name.astype('category')
  df.fancy_model_name.cat.set_categories(models_sorter, inplace=True)

  return df.sort_values(["fancy_model_name"])

def _get_pr_point_at_threshold(precision_recall_curve, thresholds):
  x = []
  y = []

  for thres in thresholds:
    pos = np.max(np.argwhere(precision_recall_curve[2] < thres))
    x.append(precision_recall_curve[1][pos])
    y.append(precision_recall_curve[0][pos])

  return x, y

def _log_prec_at_recall(precision_recall_curve, recalls):
  for rec in recalls:
    pos = np.max(np.argwhere(precision_recall_curve[1] >= rec))
    print("Recall: %f (%.2f), Precision: %f, Thereshold: %f" % \
        (precision_recall_curve[1][pos], rec, precision_recall_curve[0][pos],
         precision_recall_curve[2][pos]))

def _plot_precision_recall_curve(df,
                                 test_set_id):
  results_df = df[df.test_set_id == test_set_id]
  results_df = _sort_models_by_list(results_df, default_models_sorter)

  plt.figure(figsize=(4,4))
  count = 0
  color_list = ['darkgrey', 'black']
  linestyle_list = ['dotted', 'dashdot', 'solid']
  for _, row in results_df.iterrows():
    color = color_list[count % 2]
    linestyle = linestyle_list[count // 2]
    plt.plot(row.precision_recall_curve[1],
             row.precision_recall_curve[0],
             label=row.fancy_model_name,
             linestyle=linestyle,
             color=color,
             linewidth=1.2)
    count += 1

    if FLAGS.threshold_dots is not None:
      x, y = _get_pr_point_at_threshold(row.precision_recall_curve,
                                        FLAGS.threshold_dots)
      plt.plot(x, y, 'o', color=color)
      for i, label in enumerate(FLAGS.threshold_dots):
        plt.annotate(str(label),
                     (x[i], y[i]),
                     xytext=(x[i] + 0.01, y[i] + 0.01),
                     fontsize=8)

    if FLAGS.log_pr_auc:
      print("%s PR AUC: %.3f" % (row.fancy_model_name,
                                 auc(row.precision_recall_curve[1],
                                     row.precision_recall_curve[0])))

    if FLAGS.log_prec_at_pr is not None:
      _log_prec_at_recall(row.precision_recall_curve, FLAGS.log_prec_at_pr)

  if FLAGS.show_random_guess:
    no_skill = _calculate_no_skill_model(df, test_set_id)
    plt.plot([0, 1],
             [no_skill, no_skill],
             linestyle='--',
             color='gainsboro',
             linewidth=1.2,
             label='Random guess')
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
  else:
    legend = plt.legend(loc='lower left')
  plt.xlim(0.0, 1.00)
  plt.ylim(0.0, 1.05)
  plt.xlabel('Revocação')
  plt.ylabel('Precisão')

  for file_format in FLAGS.chart_format_list:
    chart_file_name = test_set_id + '.' + file_format
    chart_file_name = utils.get_valid_filename(chart_file_name)
    chart_file_name = os.path.join(FLAGS.charts_path, chart_file_name)

    plt.savefig(chart_file_name,
                dpi=600,
                bbox_extra_artists=(legend,),
                bbox_inches='tight')
    print('Saved chart to %s' % chart_file_name)

def _plot_pr_curve_from_files():
  results_df = eval_utils.load_results_to_df(FLAGS.results_patern)

  results_df['resolution'] = results_df.apply(
    lambda row: _get_resolution_from_ckp_name(row.model_name,row.ckpt_name),
    axis=1)

  results_df['fancy_model_name'] = results_df.apply(
      lambda row: _format_model_name(row.model_name, row.resolution), axis=1)

  results_df['test_set_id'] = results_df.apply(
      lambda row: _get_test_set_id_from_results_filename(row.file_name), axis=1)

  for test_set_id in results_df.test_set_id.unique():
    _plot_precision_recall_curve(results_df, test_set_id)

def main(_):
  _plot_pr_curve_from_files()

if __name__ == '__main__':
  app.run(main)
