from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, DepthwiseConv2D
from unit_test.helper import tf_random_seed, clean_name

model_list = []
for layer_item, kernel_size in zip([DepthwiseConv2D]*4, [(1, 1), (3, 3), (5, 5)]):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = layer_item(kernel_size)(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}_{clean_name(str(kernel_size))}')
    model_list.append(model)