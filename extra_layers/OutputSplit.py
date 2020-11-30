from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils


class OutputSplit(Layer):
    def __init__(self, count, **kwargs):
        self.count = count
        super(OutputSplit, self).__init__(**kwargs)

    def build(self, inputShape):
        super(OutputSplit, self).build(inputShape)

    def call(self, x, **kwargs):
        return [x] * self.count

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape] * self.count

    def get_config(self):
        base_config = super(OutputSplit, self).get_config()
        base_config['count'] = self.count
        return base_config

