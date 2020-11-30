from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2DTranspose
from unit_test.helper import tf_random_seed, clean_name

model_list = []
for layer_item, kernel_size in zip([Conv2DTranspose]*3, [(3, 3), (5, 5), (7, 7)]):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = layer_item(4, kernel_size, strides=(2, 2), padding='same')(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}_{clean_name(str(kernel_size))}')
    model_list.append(model)

for layer_item, kernel_size in zip([Conv2DTranspose] * 3, [(2, 2), (4, 4), (6, 6)]):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = layer_item(4, kernel_size, strides=(2, 2), padding='same')(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}_{clean_name(str(kernel_size))}')
    model_list.append(model)