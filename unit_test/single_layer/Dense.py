from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Flatten, Dense
from unit_test.helper import tf_random_seed, clean_name

model_list = []

for layer_item, units in zip([Dense]*3, [128, 64, 32]):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = Flatten()(placeholder)
    x = Dense(units, activation='relu')(x)
    model = Model(placeholder, x,
                  name=f'model_single_layer_{layer_item.__name__.lower()}_{clean_name(str(units))}')
    model_list.append(model)
