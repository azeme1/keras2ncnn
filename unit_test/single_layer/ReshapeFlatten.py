from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape
from unit_test.helper import tf_random_seed, clean_name

model_list = []
for layer_item, target_shape in zip([Reshape] * 5,
                                    [(-1, 3), (-1, 64, 3), (16, 64, 3), (32 * 32, 3), (-1, )]):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = layer_item(target_shape)(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}_{clean_name(str(target_shape))}')
    model_list.append(model)
