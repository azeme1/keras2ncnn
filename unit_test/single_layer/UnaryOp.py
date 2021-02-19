from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply
from extra_layers.UnaryOp import Sqrt
from unit_test.helper import tf_random_seed

model_list = []
for op_item in [Sqrt()]:
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = Multiply()([placeholder, placeholder])
    x = op_item(x)
    model = Model(placeholder, x, name=f'model_single_layer_{op_item.__class__.__name__.lower()}')
    model_list.append(model)
