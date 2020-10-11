from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization
from unit_test.helper import tf_random_seed, clean_name

model_list = []
for layer_item, _ in zip([BatchNormalization], [None]):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = layer_item()(placeholder)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}')
    model_list.append(model)