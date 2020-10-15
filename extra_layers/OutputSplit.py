from tensorflow.keras.layers import Layer


class OutputSplit(Layer):
    def __init__(self, count, **kwargs):
        self.count = count
        super(OutputSplit, self).__init__(**kwargs)

    def build(self, inputShape):
        super(OutputSplit, self).build(inputShape)

    def call(self, x):
        return [x] * self.count

    def compute_output_shape(self, inputShape):
        return [inputShape] * self.count

    def get_config(self):
        base_config = super(OutputSplit, self).get_config()
        base_config['count'] = self.count
        return base_config
