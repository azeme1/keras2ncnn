from tensorflow.keras.layers import Subtract
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.Div')
class Div(Subtract):
  def _merge_function(self, inputs):
    if len(inputs) != 2:
      raise ValueError('A `Div` layer should be called '
                       'on exactly 2 inputs')
    return inputs[0]/(inputs[1])