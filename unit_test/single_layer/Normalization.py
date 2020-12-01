from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization
from unit_test.helper import tf_random_seed
import numpy as np

model_list = []
for training_flag in [False, True]:
    for layer_item, _ in zip([BatchNormalization], [None]):
        tf_random_seed()
        np.random.seed(7)
        placeholder = Input((32, 32, 3), name='data')
        x = layer_item()(placeholder, training_flag)
        model = Model(placeholder, x,
                      name=f'model_single_layer_{str(training_flag)}_{layer_item.__name__.lower()}')

        model.layers[1].set_weights([np.random.uniform(0, 1, size=3),
                                     np.random.uniform(0, 1, size=3),
                                     np.random.uniform(0, 1, size=3),
                                     np.random.uniform(0, 1, size=3)])

        model_list.append(model)
