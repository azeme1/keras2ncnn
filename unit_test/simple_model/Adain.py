from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Multiply, Subtract, DepthwiseConv2D, Add, Reshape
from extra_layers.BinaryOp import Div
from extra_layers.UnaryOp import Sqrt
from tensorflow.keras.models import Model
import numpy as np

in_c = Input((None, None, 512))
in_s = Input((None, None, 512))
epsilon = 1.e-5

epsilon_tensor = DepthwiseConv2D((1, 1), weights=[np.full((1, 1, int(in_c.shape[-1]), 1), 0.),
                                 np.full((int(in_c.shape[-1]),), -epsilon)])(in_c)
zero_tensor = DepthwiseConv2D((1, 1), weights=[np.full((1, 1, int(in_c.shape[-1]), 1), 0.),
                                 np.full((int(in_c.shape[-1]),), 0)])(in_s)

mean_s = GlobalAveragePooling2D()(in_s)
diff_s = Subtract()([in_s, mean_s])
diff_square_s = Multiply()([diff_s, diff_s])
var_s = GlobalAveragePooling2D()(diff_square_s)

mean_c = GlobalAveragePooling2D()(in_c)
diff_c = Subtract()([in_c, mean_c])
diff_square_c = Multiply()([diff_c, diff_c])
var_c = GlobalAveragePooling2D()(diff_square_c)

var_c = Reshape((1, 1, int(var_c.shape[-1])))(var_c)
var_s = Reshape((1, 1, int(var_s.shape[-1])))(var_s)

var_c = Subtract()([var_c, epsilon_tensor])
var_c = Sqrt()(var_c)
var_s = Subtract()([var_s, epsilon_tensor])
var_s = Sqrt()(var_s)
mean_s = Subtract()([mean_s, epsilon_tensor])

x = Div()([diff_c, var_c])
x = Multiply()([var_s, x])
x = Add()([x, mean_s])

keras_adain_model = Model([in_c, in_s], x)

model_list = [keras_adain_model]

keras_adain_model.save('adain.hdf5')