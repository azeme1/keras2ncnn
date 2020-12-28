from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Activation, ReLU, Add
from unit_test.helper import tf_random_seed

model_list = []

for model_depth in [1]:
    tf_random_seed()
    placeholder_z = Input((32, 32, 3), name='data_1')
    placeholder_y = Input((32, 32, 3), name='data_2')
    x = Add()([placeholder_y, placeholder_z])

    model = Model([placeholder_y, placeholder_z], x, name=f'multi_input_model_{str(model_depth).zfill(3)}')
    model_list.append(model)
