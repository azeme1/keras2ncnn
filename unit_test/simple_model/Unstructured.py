from tensorflow.keras.layers import Input, Conv2D, ReLU, Concatenate
from tensorflow.keras.models import Model
from unit_test.helper import tf_random_seed

model_list = []

tf_random_seed()
# x_in = Input((32, 32, 4))
# x_0 = Conv2D(1, (3, 3))(x_in)
# x_1 = ReLU()(x_0)
# x_3 = Conv2D(4, (1, 1))(x_0)
# x_4 = Concatenate()([x_0, x_1])
# x_5 = Concatenate(axis=-1)([x_0, x_4])
# model = Model(x_in, x_5)
x_in = Input((32, 32, 3))
x_0 = Conv2D(1, (3, 3))(x_in)
s_plit_0 = (x_0)
x_1 = ReLU()(s_plit_0)
s_plit_1 = (x_1)

x_3 = Conv2D(4, (1,1))(s_plit_0)
x_4 = Concatenate(axis=-1)([s_plit_1, x_3])
x_5 = Concatenate(axis=-1)([s_plit_0, s_plit_1, x_4])

model = Model(x_in, x_5)

model_list.append(model)