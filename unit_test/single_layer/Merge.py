from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Concatenate, Add, Multiply
from unit_test.helper import tf_random_seed

model_list = []
for activation_item in ['relu', 'sigmoid', 'softmax']:
    for merge_item in [Concatenate(axis=-1), Add(), Multiply()]:
            tf_random_seed()
            placeholder = Input((32, 32, 3), name='data')
            x_0 = Activation(activation_item)(placeholder)
            x_1 = Activation(activation_item)(placeholder)
            x = merge_item([x_0, x_1])
            model = Model(placeholder, x,
                          name=f'model_single_layer_{activation_item}_{merge_item.__class__.__name__.lower()}')
            model_list.append(model)
