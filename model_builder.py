import collections

import tensorflow as tf
import tensorflow_hub as hub

class ModelSpecs(
  collections.namedtuple("ModelSpecs", [
    'uri', 'type', 'input_size', 'classes', 'activation'
  ])):
  """ The specifications for a image model
  """


def get_default_specs():
  return ModelSpecs(
    uri='https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1',
    type='tfhub',
    input_size=224,
    classes=2,
    activation='softmax'
  )

efficientnet_b0_spec = get_default_specs()._replace(
  uri='https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1',
  input_size=224
)

efficientnet_b3_spec = get_default_specs()._replace(
  uri='https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1',
  input_size=300
)

efficientnet_b4_spec = get_default_specs()._replace(
  uri='https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1',
  input_size=380
)

mobilenetv2_spec = get_default_specs()._replace(
  uri='mobilenetv2',
  type='keras',
  input_size=224
)

MODELS_SPECS = {
  'efficientnet-b0': efficientnet_b0_spec,
  'efficientnet-b3': efficientnet_b3_spec,
  'efficientnet-b4': efficientnet_b4_spec,
  'mobilenetv2': mobilenetv2_spec,
}

def _create_model_from_hub(specs):
  model = tf.keras.Sequential([
    hub.KerasLayer(specs.uri, trainable=True),
    tf.keras.layers.Dense(units=specs.classes, activation=specs.activation)
  ])

  model.build([None, specs.input_size, specs.input_size, 3])

  return model

def _create_model_from_keras(specs):
  if specs.uri == 'mobilenetv2':
    base_model = tf.keras.applications.MobileNetV2(
      input_shape=(specs.input_size, specs.input_size, 3),
      include_top=False,
      weights='imagenet'
    )
  else:
    raise RuntimeError('Model %s not implemented' % specs.uri)

  x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
  output = tf.keras.layers.Dense(specs.classes, activation=specs.activation)(x)
  model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

  return model

def _create_model_from_specs(specs):
  if specs.type == 'tfhub':
    return _create_model_from_hub(specs)
  else:
    return _create_model_from_keras(specs)

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
