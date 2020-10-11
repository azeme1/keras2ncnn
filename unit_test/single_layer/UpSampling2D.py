from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D
from unit_test.helper import tf_random_seed, clean_name

model_list = []
for layer_item, size_item, interpolation in zip([UpSampling2D]*4,
                                 [(2, 2), (4, 4), (2, 2), (4, 4)],
                                 ['nearest', 'nearest', 'bilinear', 'bilinear']):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = layer_item(size_item, interpolation=interpolation)(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}_{clean_name(str(size_item))}_{interpolation}')
    model_list.append(model)

