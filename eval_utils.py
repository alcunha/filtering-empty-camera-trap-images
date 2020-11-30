import os

import collections
import pickle
import tensorflow as tf
import pandas as pd

ClassifierResults = collections.namedtuple("ClassifierResults", [
    'model_name', 'ckpt_name', 'accuracy', 'confusion_matrix',
    'precision_recall_f1_per_class', 'precision_recall_curve'
  ])


def save_results_to_file(filename,
                         model_name,
                         ckpt_name,
                         accuracy,
                         confusion_matrix,
                         precision_recall_f1_per_class,
                         precision_recall_curve):

  results = ClassifierResults(model_name,
                              ckpt_name,
                              accuracy,
                              confusion_matrix,
                              precision_recall_f1_per_class,
                              precision_recall_curve)

  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  with open(filename, 'wb') as file_obj:
    pickle.dump(results, file_obj)

  print('Saved restuls to %s' % filename)

  return results

def load_results_from_file(filename):

  with open(filename, 'rb') as file_obj:
    results = pickle.load(file_obj)

  return results

def load_results_to_df(results_patern):
  file_list = tf.io.gfile.glob(results_patern)
  results_list = []
  
  for file_result in file_list:
    result_dict = load_results_from_file(file_result)._asdict()
    result_dict['file_name'] = file_result
    results_list.append(result_dict)
  
  df = pd.DataFrame(results_list)
  
  return df
