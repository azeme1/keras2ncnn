from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, ReLU, LeakyReLU
from unit_test.helper import tf_random_seed

model_list = []
for activation_item in ['relu', 'softmax', 'sigmoid']:
    tf_random_seed()
    placeholder = Input((32,32,3), name='data')
    x = Activation(activation_item)(placeholder)
    model = Model(placeholder, x, name=f'model_single_layer_{activation_item}')
    model_list.append(model)

for op_item in [ReLU(), LeakyReLU(0.1), LeakyReLU(0.25)]:
    tf_random_seed()
    placeholder = Input((32,32,3), name='data')
    x = op_item(placeholder)
    model = Model(placeholder, x, name=f'model_single_layer_{op_item.__class__.__name__.lower()}')
    model_list.append(model)