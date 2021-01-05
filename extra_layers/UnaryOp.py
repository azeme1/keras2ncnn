import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras import backend as K
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.RSqrt')
class RSqrt(ReLU):
  def call(self, inputs):
    return tf.rsqrt(inputs)


@tf_export('keras.layers.Sqrt')
class Sqrt(ReLU):
  def call(self, inputs):
    return K.sqrt(inputs)