import collections
import pickle

ClassifierResults = collections.namedtuple("ClassifierResults", [
    'model_name', 'ckpt_name', 'accuracy', 'confusion_matrix',
    'precision_recall', 'precision_recall_curve'
  ])


def save_results_to_file(filename,
                         model_name,
                         ckpt_name,
                         accuracy,
                         confusion_matrix,
                         precision_recall,
                         precision_recall_curve):

  results = ClassifierResults(model_name,
                              ckpt_name,
                              accuracy,
                              confusion_matrix,
                              precision_recall,
                              precision_recall_curve)

  with open(filename, 'wb') as file_obj:
    pickle.dump(results, file_obj)

  print('Saved restuls to %s' % filename)

  return results

def load_results_from_file(filename):

  with open(filename, 'rb') as file_obj:
    results = pickle.load(file_obj)

  return results
