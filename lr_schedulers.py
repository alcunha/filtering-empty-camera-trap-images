import math

import tensorflow as tf

def lr_cosine_decay(initial_learning_rate,
                    current_step,
                    decay_steps,
                    alpha=0.0):

  if current_step > decay_steps:
    current_step = decay_steps

  cosine_decay = 0.5 * (1 + tf.math.cos(
                                  math.pi * current_step / float(decay_steps)))
  decayed = (1 - alpha) * cosine_decay + alpha

  return initial_learning_rate * decayed

def lr_linear_warmup(initial_learning_rate, current_step, warmup_steps):

  return current_step * initial_learning_rate / float(warmup_steps)

class CosineDecayWithLinearWarmUpScheduler(tf.keras.callbacks.Callback):

  def __init__(self,
               initial_learning_rate,
               decay_steps,
               warmup_steps=0,
               alpha=0.0):

    super(CosineDecayWithLinearWarmUpScheduler, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.warmup_steps = warmup_steps
    self.alpha = alpha
    self.steps = 0
    self.learning_rates = []

  def on_train_batch_begin(self, batch, logs=None):
    if not hasattr(self.model.optimizer, "lr"):
      raise ValueError('Optimizer must have a "lr" attribute.')

    self.steps = self.steps + 1
    
    if self.steps < self.warmup_steps:
      lr = lr_linear_warmup(
              self.initial_learning_rate,
              self.steps,
              self.warmup_steps)
    else:            
      lr = lr_cosine_decay(
              self.initial_learning_rate,
              self.steps - self.warmup_steps,
              self.decay_steps,
              self.alpha)
    
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
    
    self.learning_rates.append(lr)
