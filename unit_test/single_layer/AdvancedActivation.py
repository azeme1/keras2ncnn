from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from unit_test.helper import tf_random_seed


# Copy from Keras Applications MobileNetV3
def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


model_list = []
for activation, activation_name in zip([hard_sigmoid, hard_swish], ['hard_sigmoid', 'hard_swish']):
    tf_random_seed()
    placeholder = Input((32, 32, 3), name='data')
    x = relu(placeholder)
    x = activation(x)
    model = Model(placeholder, x, name=f'model_single_layer_{activation_name}')
    model_list.append(model)
