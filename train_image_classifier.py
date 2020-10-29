import os
import collections

import tensorflow as tf

import lr_schedulers

class HParams(
  collections.namedtuple("HParams", [
    'lr', 'use_cosine_decay', 'warmup_steps', 'epochs', 'batch_size',
    'momentum', 'use_label_smoothing', 'use_logits', 'model_dir'
  ])):
  """ The hyperparams for training a model
  """

def get_default_hparams():
  return HParams(
    lr=0.01,
    use_cosine_decay=False,
    warmup_steps=500,
    epochs=10,
    batch_size=32,
    momentum=0.0,
    use_label_smoothing=False,
    use_logits=False,
    model_dir='/tmp/models/'
  )

def generate_optimizer(hparams):
  optimizer = tf.keras.optimizers.SGD(lr=hparams.lr, momentum=hparams.momentum)

  return optimizer

def generate_loss_fn(hparams):
  loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=hparams.use_logits,
    label_smoothing=hparams.use_label_smoothing
    )

  return loss_fn

def generate_lr_scheduler(hparams, steps_per_epoch):
  scheduler = lr_schedulers.CosineDecayWithLinearWarmUpScheduler(
    initial_learning_rate=hparams.lr,
    decay_steps=hparams.epochs*steps_per_epoch,
    warmup_steps=hparams.warmup_steps,
  )

  return scheduler

def train_model(model, hparams, train_data_and_size, val_data_and_size):

  train_data, train_size = train_data_and_size
  val_data, val_size = val_data_and_size

  steps_per_epoch = train_size // hparams.batch_size
  validation_steps = val_size // hparams.batch_size

  summary_dir = os.path.join(hparams.model_dir, "summaries")
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)

  checkpoint_filepath = os.path.join(hparams.model_dir, "ckp")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      save_freq='epoch')

  callbacks = [summary_callback, checkpoint_callback]

  if val_data is not None:
    best_model_filepath = os.path.join(hparams.model_dir, 'best_model', 'ckp')
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    callbacks.append(best_model_callback)

  if hparams.use_cosine_decay:
    callbacks.append(generate_lr_scheduler(hparams, steps_per_epoch))

  optimizer = generate_optimizer(hparams)
  loss_fn = generate_loss_fn(hparams)

  model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

  return model.fit(
    train_data,
    epochs=hparams.epochs,
    callbacks=callbacks,
    validation_data=val_data,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
  )
