import tensorflow as tf
from tensorflow.python.util.tf_export import tf_export
from tensorflow.keras.layers import ZeroPadding2D


@tf_export('keras.layers.ZeroPadding2D')
class ReflectPadding2D(ZeroPadding2D):
  def __init__(self, padding=(1, 1), data_format=None, **kwargs):
    super(ReflectPadding2D, self).__init__(**kwargs)

  def call(self, inputs):
      padding = self.padding
      data_format = self.data_format
      assert len(padding) == 2
      assert len(padding[0]) == 2
      assert len(padding[1]) == 2
      if data_format is None:
          data_format = 'channels_last'
      if data_format not in {'channels_first', 'channels_last'}:
          raise ValueError('Unknown data_format: ' + str(data_format))

      if data_format == 'channels_first':
          pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
      else:
          pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
      return tf.pad(inputs, pattern, mode='REFLECT')

