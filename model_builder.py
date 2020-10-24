import collections

import tensorflow as tf
import tensorflow_hub as hub

class ModelSpecs(
  collections.namedtuple("ModelSpecs", [
    'uri', 'input_size', 'classes', 'activation'
  ])):
  """ The specifications for a image model
  """


def get_default_specs():
  return ModelSpecs(
    uri='https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1',
    input_size=224,
    classes=2,
    activation='softmax'
  )

efficientnet_b0_spec = get_default_specs()._replace(
  uri='https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1',
  input_size=224
)

efficientnet_lite0_spec = get_default_specs()._replace(
  uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
  input_size=224
)

MODELS_SPECS = {
  'efficientnet-b0': efficientnet_b0_spec,
  'efficientnet-lite0': efficientnet_lite0_spec,
}

def _create_model_from_specs(specs):
  model = tf.keras.Sequential([
    hub.KerasLayer(specs.uri, trainable=True),
    tf.keras.layers.Dense(units=specs.classes, activation=specs.activation)
  ])

  model.build([None, specs.input_size, specs.input_size, 3])

  return model


def create(model_name,
           num_classes,
           input_size=None,
           classifier_activation="softmax"):

  if model_name not in MODELS_SPECS.keys():
    raise RuntimeError('Model %s not implemented' % model_name)

  specs = MODELS_SPECS[model_name]
  specs = specs._replace(
    classes=num_classes,
    activation=classifier_activation,
  )
  if input_size is not None:
    specs = specs._replace(input_size=input_size)

  return _create_model_from_specs(specs)
