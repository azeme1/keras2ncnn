from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, AveragePooling2D, AvgPool2D, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAvgPool2D, GlobalMaxPooling2D, GlobalMaxPool2D
from unit_test.helper import tf_random_seed, clean_name

model_list = []
for layer_item, pool_item in zip([MaxPooling2D, AveragePooling2D, MaxPool2D, AvgPool2D],
                                 [2, (2, 2), 4, (4, 4)]):
    tf_random_seed()
    placeholder = Input((32,32,3), name='data')
    op_item = layer_item(pool_item)
    x = op_item(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{op_item.__class__.__name__.lower()}_{clean_name(str(pool_item))}')
    model_list.append(model)

for layer_item in [GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPool2D, GlobalAvgPool2D]:
    tf_random_seed()
    placeholder = Input((32,32,3), name='data')
    op_item = layer_item()
    x = op_item(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{op_item.__class__.__name__.lower()}')
    model_list.append(model)
