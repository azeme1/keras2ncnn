import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import ncnn
import numpy as np

from converter.converter import conver_model
from converter.model_adaptation import adapt_keras_model, convert_blob, clean_node_name
from optimization.optimize_graph import apply_transformations, check_transform
from unit_test.helper import save_config
# from unit_test.single_layer.Activation import model_list
# from unit_test.single_layer.Conv2D import model_list
# from unit_test.single_layer.DepthwiseConv2D import model_list
# from unit_test.single_layer.Normalization import model_list
# from unit_test.single_layer.Pooling2D import model_list
# from unit_test.single_layer.UpSampling2D import model_list
from unit_test.single_layer.ReshapeFlatten import model_list
# from unit_test.single_layer.Merge import model_list
# from unit_test.single_layer.Conv2DTranspose import model_list
# from unit_test.simple_model.EncoderDecoder import model_list
# from unit_test.simple_model.UNet import model_list
# model_list = [load_model('./model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5')]
# model_list = [load_model('./model_zoo/style_transfer/pix2pix/cats_model_small.hdf5')]


def mat_to_numpy_4(mat_array):
    np_array = np.array(mat_array)
    assert len(np_array.shape) == 3, f"Wrong Array Shape {np_array.shape}"
    np_array = np_array.reshape((1,) + np_array.shape)
    return np_array


def mat_to_numpy_3(mat_array):
    np_array = np.array(mat_array)
    assert len(np_array.shape) == 2, f"Wrong Array Shape {np_array.shape}"
    np_array = np_array.reshape((1,) + np_array.shape)
    return np_array


def mat_to_numpy_2(mat_array):
    np_array = np.array(mat_array)
    assert len(np_array.shape) == 1, f"Wrong Array Shape {np_array.shape}"
    np_array = np_array.reshape((1,) + np_array.shape)
    return np_array


def tensor_nchw2nhwc_4(in_data):
    return np.transpose(in_data, (0, 2, 3, 1))


def tensor_nchw2nhwc_3(in_data):
    return np.transpose(in_data, (0, 2, 1))


def tensor_nchw2nhwc_2(in_data):
    return in_data


def tensor4_ncnn2keras(mat_array):
    if mat_array.dims == 3:
        return tensor_nchw2nhwc_4(mat_to_numpy_4(mat_array))
    elif mat_array.dims == 2:
        return tensor_nchw2nhwc_3(mat_to_numpy_3(mat_array))
    elif mat_array.dims == 1:
        return tensor_nchw2nhwc_2(mat_to_numpy_2(mat_array))
    else:
        raise NotImplemented


export_root = './unit_test_output/'

for keras_model_in in model_list:
    # keras_model_in.summary()
    keras_model = apply_transformations(keras_model_in)
    adapted_keras_model = adapt_keras_model(keras_model, keras_model.name)
    check_transform(keras_model_in, adapted_keras_model, False)
    string_list, weight_list, layer_name_list = conver_model(adapted_keras_model, False, False)

    export_path = os.path.join(export_root, '', keras_model.name)
    os.makedirs(export_path, exist_ok=True)
    out_config_path, out_weights_path = save_config(string_list, weight_list, adapted_keras_model.name, export_path,
                                                    debug=False)

    target_shape = (1,) + keras_model.input_shape[1:]
    src_x, src_y = target_x, target_y = target_shape[1:3]

    frame = np.random.uniform(0, 255, size=target_shape[1:]).astype(np.uint8)
    mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR, src_x, src_y, target_x, target_y)

    mean = np.array([128] * 3)
    std = 1. / np.array([128] * 3)
    mat_in.substract_mean_normalize(mean, std)

    # Check input
    keras_tensor_cmp = tensor4_ncnn2keras(mat_in)
    keras_tensor = keras_in = (frame[None, ...] - mean) * std
    assert np.abs(keras_tensor - keras_tensor_cmp).sum() < 1.e-5, 'Bad Input Tensor!'

    net = ncnn.Net()
    net.load_param(out_config_path)
    net.load_model(out_weights_path)

    num_threads = 4
    error_th = 1.e-5
    ex = net.create_extractor()
    ex.set_num_threads(num_threads)

    assert len(adapted_keras_model.inputs) == 1, "MultiInput is not supported!"

    ncnn_input_name = clean_node_name(adapted_keras_model.inputs[0].name)
    ex.input(ncnn_input_name, mat_in)
    mat_out = ncnn.Mat()

    print('\n' + ('=' * 20) + 'By Layer Comparison ' + ('=' * 20))
    for layer in adapted_keras_model.layers:
        layer_name = layer.name
        layer_output = convert_blob(layer.output)

        test_keras_model = K.function(adapted_keras_model.inputs, adapted_keras_model.get_layer(layer_name).output)
        keras_output = test_keras_model(keras_in)

        for item, tensor_true in zip(layer_output, convert_blob(keras_output)):

            tensor_name = clean_node_name(item.name)
            ex.extract(tensor_name, mat_out)

            tensor_exp = tensor4_ncnn2keras(mat_out)
            error_exp = np.abs(tensor_true - tensor_exp).mean()
            print(f'Layer - {layer_name} :: {error_exp} < {str(error_th)} {error_exp < error_th}')
            ...