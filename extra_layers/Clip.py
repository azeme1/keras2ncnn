from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Clip(Layer):
    def __init__(self, min_value=-1, max_value=+1, **kwargs):
        super(Clip, self).__init__(**kwargs)
        self.min_value = K.cast_to_floatx(min_value)
        self.max_value = K.cast_to_floatx(max_value)

    def call(self, inputs):
        return K.clip(inputs, min_value=self.min_value, max_value=self.max_value)

    def get_config(self):
        config = {
            'min_value': self.min_value,
            'max_value': self.max_value,
        }
        base_config = super(Clip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
