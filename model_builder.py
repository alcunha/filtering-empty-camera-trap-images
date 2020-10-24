import tensorflow as tf
import tensorflow_hub as hub

efficientnet_b0_spec = lambda: dict(
  uri='https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1',
  input_size=224
)

efficientnet_lite0_spec = lambda: dict(
  uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
  input_size=224
)

MODELS_SPECS = {
  'efficientnet-b0': efficientnet_b0_spec,
  'efficientnet-lite0': efficientnet_lite0_spec,
}

def _create_model_from_specs(specs):
  model = tf.keras.Sequential([
    hub.KerasLayer(specs['uri'], trainable=True),
    tf.keras.layers.Dense(units=specs['classes'],
                          activation=specs['activation'])
  ])

  model.build([None, specs['input_size'], specs['input_size'], 3])

  return model


def create(model_name,
           num_classes,
           input_size=None,
           classifier_activation="softmax"):

  if model_name not in MODELS_SPECS.keys():
    raise RuntimeError('Model %s not implemented' % model_name)

  specs = MODELS_SPECS[model_name]()
  custom_specs = dict(classes=num_classes,
                      activation=classifier_activation)
  specs.update(custom_specs)
  if input_size is not None:
    specs.update({'input_size': input_size})
  
  return _create_model_from_specs(specs)
