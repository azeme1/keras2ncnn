from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Activation, ReLU
from unit_test.helper import tf_random_seed

model_list = []

for model_depth in [1, 2]:
    tf_random_seed()
    placeholder = x = Input((32, 32, 3), name='data')
    for j in range(model_depth):
        x = Conv2D(8, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2,2))(x)

    for j in range(model_depth):
        x = Conv2D(8, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(1, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(placeholder, x, name=f'simple_model_encoder_decoder_{str(model_depth).zfill(3)}')
    model_list.append(model)

