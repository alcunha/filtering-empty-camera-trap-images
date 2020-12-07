import os

import collections
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd

import model_builder
import train_image_classifier

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

def _generate_fake_instance(input_size, num_classes):
  instance_shape = (input_size, input_size, 3)
  fake_instance = np.ones(instance_shape, dtype=np.float32)
  fake_labels = np.ones((num_classes,))

  return fake_instance, fake_labels

def _initialize_model_optimizer(model, input_size, num_classes):
  fake_steps = 2
  fake_instance, fake_labels = _generate_fake_instance(input_size, num_classes)
  x = np.array([fake_instance for i in range(fake_steps)])
  y = np.array([fake_labels for i in range(fake_steps)])

  model.fit(
    x, y, batch_size=1, epochs=1, steps_per_epoch=fake_steps)

def load_model_from_checkpoint(model_name, num_classes, input_size, ckpt_dir):
  model = model_builder.create(
    model_name=model_name,
    num_classes=num_classes,
    input_size=input_size
  )

  hparams = train_image_classifier.get_default_hparams()
  optimizer = train_image_classifier.generate_optimizer(hparams)
  loss_fn = train_image_classifier.generate_loss_fn(hparams)
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

  # workaround to fix 'Unresolved object in checkpoint' for optimizer variables
  _initialize_model_optimizer(model, input_size, num_classes)

  checkpoint_path = os.path.join(ckpt_dir, "ckp")

  model.load_weights(checkpoint_path)

  return model
