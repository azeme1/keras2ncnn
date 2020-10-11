from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Activation, \
                                    ReLU, Concatenate, LeakyReLU, Add
from unit_test.helper import tf_random_seed

model_list = []

for model_depth in [1, 2]:
    tf_random_seed()
    join_list = []
    placeholder = x = Input((32, 32, 3), name='data')
    for j in range(model_depth):
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        join_list.append(x)
        x = MaxPooling2D((2,2))(x)

    join_list = list(reversed(join_list))
    for j in range(model_depth):
        x = Conv2D(8, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2,2), interpolation='bilinear')(x)
        x = Concatenate(axis=-1)([x, join_list[j]])

    x = Conv2D(1, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(placeholder, x, name=f'simple_model_unet_{str(model_depth).zfill(3)}')
    model_list.append(model)

filter_count_list = [12, 16, 32, 64, 128, 128 + 64, 256]
kernel_size_list = [(3, 3)] * (len(filter_count_list) - 1) + [(4, 4)]
iter_count_list = [2] * 7
in_layer = c_layer = Input((256, 256, 3), name='data')
upsamplig_type = 'nearest'  # 'bilinear'

join_layer_list = []
for i in range(len(filter_count_list) - 1):
    for j in range(iter_count_list[i]):
        c_layer = Conv2D(filter_count_list[i], kernel_size_list[i], padding='same')(c_layer)
        #         c_layer = BatchNormalization()(c_layer)
        #         c_layer = ReLU()(c_layer)
        c_layer = LeakyReLU(0.1)(c_layer)
    join_layer_list.append(c_layer)
    c_layer = MaxPooling2D((2, 2))(c_layer)
    # print(c_layer, kernel_size_list[i])

join_layer_list = list(reversed(join_layer_list))
filter_count_list = list(reversed(filter_count_list[:-1]))
kernel_size_list = list(reversed(kernel_size_list[:-1]))
for i in range(len(filter_count_list)):
    c_layer = UpSampling2D((2, 2), interpolation=upsamplig_type)(c_layer)
    c_layer = Concatenate(axis=-1)([c_layer, join_layer_list[i]])
    for j in range(iter_count_list[i]):
        c_layer = Conv2D(filter_count_list[i], kernel_size_list[i], padding='same')(c_layer)
        #   c_layer = BatchNormalization()(c_layer)
        #   c_layer = ReLU()(c_layer)
        c_layer = LeakyReLU(0.1)(c_layer)
    # print(c_layer, kernel_size_list[i])

c_layer = Conv2D(1, (5, 5), padding='same')(c_layer)
# c_layer = BatchNormalization()(c_layer)
c_layer = Activation('sigmoid')(c_layer)

model = Model(in_layer, c_layer, name='selfie_photo_segmentation')
model_list.append(model)