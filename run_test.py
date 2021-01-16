import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import ncnn
import numpy as np
import cv2

from converter.converter import conver_model
from converter.model_adaptation import adapt_keras_model, convert_blob, clean_node_name
from optimization.optimize_graph import apply_transformations, check_transform
from unit_test.helper import save_config
from extra_layers.CustomObjects import extra_custom_objects
from unit_test.helper import fix_none_in_shape
# from unit_test.single_layer.Activation import model_list
# from unit_test.single_layer.Conv2D import model_list
# from unit_test.single_layer.DepthwiseConv2D import model_list
# from unit_test.single_layer.Normalization import model_list
# from unit_test.single_layer.Pooling2D import model_list
# from unit_test.single_layer.UpSampling2D import model_list
# from unit_test.single_layer.ReshapeFlatten import model_list
# from unit_test.single_layer.Dense import model_list
# from unit_test.single_layer.Merge import model_list
# from unit_test.single_layer.Conv2DTranspose import model_list
# from unit_test.single_layer.UnaryOp import model_list
# from unit_test.single_layer.BinaryOp import model_list

# from unit_test.simple_model.EncoderDecoder import model_list
# from unit_test.simple_model.UNet import model_list
# from unit_test.simple_model.MultipleInput import model_list
# from unit_test.simple_model.Adain import model_list

# model_list = [load_model('unit_test_output/encoder.hdf5', custom_objects=extra_custom_objects),
#               load_model('unit_test_output/decoder.hdf5', custom_objects=extra_custom_objects),
#               load_model('unit_test_output/adain.hdf5', custom_objects=extra_custom_objects),
#               ]
# model_list = [load_model('unit_test_output/keras_arbitrary_style_transfer.hdf5', custom_objects=extra_custom_objects)]
# model_list = [load_model('C:/Users/olga/projects/model_zoo/model_keras_ready/divamgupta/image-segmentation-keras/_adaptation/pspnet_50_ADE_20K.hdf5',
#                          custom_objects=extra_custom_objects)]
# model_list = [load_model('C:/Users/olga/projects/model_zoo/model_keras_ready/divamgupta/image-segmentation-keras/_adaptation/pspnet_101_voc12.hdf5',
#                          custom_objects=extra_custom_objects)]
# model_list = [load_model('model_privat/style_transfer/pix2pix/cats_v1.hdf5')] # code demo
# model_list = [load_model('model_zoo/detection/AIZOOTech_I_FaceMaskDetection/face_mask_detection_optimized.hdf5')] #issue 1
# model_list = [load_model('./model_zoo/variouse/issue_00003/fiop_dumb_model_fixed.h5')] #issue 3
# model_list = [load_model('model_zoo/variouse/issue_00006/deconv_fin_munet.h5')] #issue 6

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
        print(f'tensor4_ncnn2keras :: {mat_array.dims}')
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

    net = ncnn.Net()
    net.load_param(out_config_path)
    net.load_model(out_weights_path)

    num_threads = 4
    error_th = 1.e-5
    ex = net.create_extractor()
    ex.set_num_threads(num_threads)

    if type(keras_model.input_shape) != list:
        target_shape_list = [keras_model.input_shape]
    else:
        target_shape_list = keras_model.input_shape

    keras_in_list = []
    for input_index, target_shape in enumerate(target_shape_list):
        target_shape = fix_none_in_shape(target_shape)
        src_x, src_y = target_x, target_y = target_shape[1:3]

        if target_shape[-1] == 3:
            frame = np.random.uniform(0, 255, size=fix_none_in_shape(target_shape)[1:]).astype(np.uint8)
            mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR, src_x, src_y, target_x, target_y)
            mean = np.array([0] * 3)
            std = 1. / np.array([255.] * 3)
            mat_in.substract_mean_normalize(mean, std)
            keras_tensor = (frame[None, ...] - mean) * std
        else:
            frame = np.random.uniform(0., +1., size=fix_none_in_shape(target_shape))
            mat_in = ncnn.Mat(np.transpose(frame, (0, 3, 1, 2))[0].astype(np.float32))
            keras_tensor = np.transpose(np.array(mat_in)[None, ...], (0, 2, 3, 1))


        # Check input
        keras_tensor_cmp = tensor4_ncnn2keras(mat_in)
        keras_in_list.append(keras_tensor)
        assert np.abs(keras_tensor - keras_tensor_cmp).mean() < 1.e-5, 'Bad Input Tensor!'

        ncnn_input_name = clean_node_name(adapted_keras_model.inputs[input_index].name)
        ex.input(ncnn_input_name, mat_in)

    print('\n' + ('=' * 20) + 'Test mode from ./run_test.py' + ('=' * 20))
    print('\n' + ('=' * 20) + 'By Layer Comparison ' + ('=' * 20))
    mat_out = ncnn.Mat()
    inference_sum = 0
    for layer in adapted_keras_model.layers:
        layer_name = layer.name
        layer_output = convert_blob(layer.output)

        test_keras_model = K.function(adapted_keras_model.inputs, adapted_keras_model.get_layer(layer_name).output)
        keras_output = test_keras_model(keras_in_list)

        for item, tensor_true in zip(layer_output, convert_blob(keras_output)):

            tensor_name = clean_node_name(item.name)
            ex.extract(tensor_name, mat_out)

            tensor_exp = tensor4_ncnn2keras(mat_out)
            inference_sum += np.prod(tensor_true.shape)
            try:
                error_exp = np.abs(tensor_true - tensor_exp).mean()
                print(f'Layer - {layer_name} inference MAE :: {error_exp} < {str(error_th)} {error_exp < error_th} ' +
                      f"Keras::{tensor_true.shape} / NCNN::{tensor_exp.shape}")
            except Exception as e_item:
                print('-' * 10, layer_name, '-' * 10)
                print(layer_name, str(e_item))
            ...

    inference_sum_float32 = int(0.5 + (inference_sum*4.)/(2**20))
    print(f'Estimated float32 inference memory :: {inference_sum_float32} MB')
