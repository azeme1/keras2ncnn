from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Subtract, DepthwiseConv2D
from extra_layers.BinaryOp import Div
from unit_test.helper import tf_random_seed, np_random_seed
import numpy as np

model_list = []
for op_item in [Subtract(), Div()]:
    epsilon = 1.e-5
    tf_random_seed()
    np_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = placeholder
    y = DepthwiseConv2D((1, 1), weights=[np.random.uniform(+1, +2, size=(1, 1, int(x.shape[-1]), 1)),
                                         np.full((int(x.shape[-1]),), epsilon)])(x)
    x = op_item([x, y])
    model = Model(placeholder, x, name=f'model_single_layer_{op_item.__class__.__name__.lower()}')
    model_list.append(model)
