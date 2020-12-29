from tensorflow.keras.layers import Subtract
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.Div')
class Div(Subtract):
  def _merge_function(self, inputs):
    if len(inputs) != 2:
      raise ValueError('A `Subtract` layer should be called '
                       'on exactly 2 inputs')
    epsilon = 1e-5
    return inputs[0]/(inputs[1] + epsilon)