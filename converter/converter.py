# https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step
# https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
# https://github.com/Tencent/ncnn/wiki/operation-param-weight-table

import numpy as np
from tqdm import tqdm
from functools import partial
from collections import OrderedDict

from converter.model_adaptation import get_outbound_nodes, convert_blob, clean_node_name

activation_type_dict = {'linear': 0, 'relu': 1, 'sigmoid': 4, 'tanh': 5}

layer_type_mapping = {'OutputSplit': 'Split', 'InputLayer': 'Input', 'ReLU': 'ReLU', 'LeakyReLU': 'ReLU',
                      'MaxPooling2D': 'Pooling', 'AveragePooling2D': 'Pooling',
                      'MaxPool2D': 'Pooling', 'AvgPool2D': 'Pooling',
                      'GlobalMaxPooling2D': 'Pooling', 'GlobalAveragePooling2D': 'Pooling',
                      'GlobalMaxPool2D': 'Pooling', 'GlobalAvgPool2D': 'Pooling',
                      'Conv2D': 'Convolution', 'Concatenate': 'Concat',
                      'UpSampling2D': 'Interp', 'Add': 'Eltwise', 'Multiply': 'Eltwise',
                      'DepthwiseConv2D': 'ConvolutionDepthWise', 'BatchNormalization': 'BatchNorm',
                      'Conv2DTranspose': 'Deconvolution', 'ZeroPadding2D': 'Padding', 'ReflectPadding2D': 'Padding',
                      'Reshape': 'Reshape',
                      'Clip': 'Clip', 'InstanceNormalization': 'InstanceNorm',
                      'sigmoid': 'Sigmoid', 'softmax': 'Softmax', 'relu': 'ReLU', 'tanh': 'TanH', 'Flatten': 'Reshape',
                      'Dense': 'InnerProduct',
                      'Sqrt': 'UnaryOp'}


def fix_axis_value(in_dict, axis):
    if axis < 0:
        axis = len(in_dict['layer'].output_shape) + axis

    if len(in_dict['layer'].output_shape) == 4:
        magic_number = 2
    elif len(in_dict['layer'].output_shape) == 3:
        magic_number = 1
    else:
        raise NotImplemented

    # Need this NCNN has 3-dim tensors
    if 'NCHW':
        axis = magic_number - int(axis - 1)
    else:
        assert False, 'NHWC is not supported'
    assert axis >= 0, 'Axis can not be negative'
    return axis


def get_valid_shape(raw_shape):
    if isinstance(raw_shape, list):
        layer_shape_list = raw_shape
    else:
        layer_shape_list = [raw_shape]
    return layer_shape_list


def get_layer_type(layer):
    type_mapping = type(layer).__name__
    if type_mapping == 'Activation':
        type_mapping = layer.get_config()['activation']
    elif type_mapping == 'BatchNormalization':
        try:
            call_args = layer.inbound_nodes[0].call_kwargs
            if 'training' in call_args:
                if call_args['training'] in [1, True]:
                    type_mapping = 'InstanceNormalization'
        except Exception as e_item:
            print('Implicit Instance Normalization')
            print(str(e_item))
    mapping_function_name = '_'.join(['get', str(type_mapping).lower(), 'mapping'])
    return layer_type_mapping[type_mapping], mapping_function_name


def get_layer_name(layer):
    return layer.get_config()['name']


def get_blob_shape_string(layer, batch_size):
    N = batch_size
    item_counter = 0
    item_list = []
    layer_output_shape = get_valid_shape(layer.output_shape)
    for output_shape in layer_output_shape:
        item_counter += 4
        item_list.extend((N,) + output_shape[1:])
    blob_shape_string = ','.join([str(item) for item in [item_counter] + item_list])
    return blob_shape_string


def get_split_shape_string(layer_output_shape, batch_size):
    N = batch_size
    item_counter = 0
    item_list = []
    for output_shape in layer_output_shape:
        _, H, W, C = output_shape
        item_counter += 4
        item_list.extend([N, H, W, C])
    blob_shape_string = ','.join([str(item) for item in [item_counter] + item_list])
    return blob_shape_string


def split_remap(blob_name, split_info):
    if blob_name in split_info:
        out_blob_name = split_info[blob_name]
    else:
        out_blob_name = blob_name
    return out_blob_name


def get_in_out_string(in_dict):
    layer = in_dict['layer']
    input_blobs = []
    output_blobs = []

    layer_input = convert_blob(layer.input)
    if layer.__class__.__name__ == 'InputLayer':
        layer_input = []
    layer_output = convert_blob(layer.output)

    for item in layer_input:
        input_blobs.append(clean_node_name(item.name))

    for item in layer_output:
        output_blobs.append(clean_node_name(item.name))

    in_out_list = [str(len(input_blobs)), str(len(output_blobs)), ' '.join(input_blobs), ' '.join(output_blobs)]
    in_out_string = ' '.join(in_out_list)

    split_string = None
    return in_out_string, split_string, (input_blobs + output_blobs)


def get_outputsplit_mapping(in_dict):
    parameter_mapping = OrderedDict({})
    return parameter_mapping


def get_reshape_mapping(in_dict):
    # Reshape     0	w	-233
    #             1	h	-233
    #             2	c	-233
    #             3	permute	0
    layer_config = in_dict['layer'].get_config()
    target_shape = layer_config['target_shape']
    if len(target_shape) == 3:
        y_size, x_size, c_size = target_shape
        parameter_mapping = OrderedDict({0: x_size, 1: y_size, 2: c_size, 3: 0})
    elif len(target_shape) == 2:
        xy_size, c_size = target_shape
        parameter_mapping = OrderedDict({0: xy_size, 1: c_size, 2: -233, 3: 1})
    elif len(target_shape) == 1:
        c_size, = target_shape
        parameter_mapping = OrderedDict({0: c_size, 1: -233, 2: -233, 3: 1})
    else:
        raise NotImplemented
    return parameter_mapping


def get_flatten_mapping(in_dict):
    # Reshape     0	w	-233
    #             1	h	-233
    #             2	c	-233
    #             3	permute	0
    parameter_mapping = OrderedDict({0: -1, 3: 1})
    return parameter_mapping


def get_unaryop_mapping(in_dict, optype):
    # enum OperationType
    # {
    #     Operation_ABS = 0,
    #     Operation_NEG = 1,
    #     Operation_FLOOR = 2,
    #     Operation_CEIL = 3,
    #     Operation_SQUARE = 4,
    #     Operation_SQRT = 5,
    #     Operation_RSQRT = 6,
    #     Operation_EXP = 7,
    #     Operation_LOG = 8,
    #     Operation_SIN = 9,
    #     Operation_COS = 10,
    #     Operation_TAN = 11,
    #     Operation_ASIN = 12,
    #     Operation_ACOS = 13,
    #     Operation_ATAN = 14,
    #     Operation_RECIPROCAL = 15,
    #     Operation_TANH = 16
    # };
    parameter_mapping = OrderedDict({0: optype})
    return parameter_mapping

get_sqrt_mapping = partial(get_unaryop_mapping, optype=5)


def get_inputlayer_mapping(in_dict):
    # Input	0   w	0
    #       1   h	0
    #       2	c	0
    layer_config = in_dict['layer'].get_config()
    N, H, W, C = layer_config['batch_input_shape']
    parameter_mapping = OrderedDict()
    if W is not None:
        parameter_mapping[0] = W
    if H is not None:
        parameter_mapping[1] = H
    if C is not None:
        parameter_mapping[2] = C
    return parameter_mapping


def get_padding_mapping(in_dict, pad_type=0):
    #   Padding
    #   0	top	0
    #	1	bottom	0
    #   2	left	0
    #   3	right	0
    #   4	type	0
    #   5	value	0.f
    #   6	per_channel_pad_data_size	0
    #   7	front	0
    #   8	behind	0
    # int type; -> pad_type // 0=CONSTANT 1=REPLICATE 2=REFLECT
    layer_config = in_dict['layer'].get_config()
    per_channel_pad_data_size = 0
    front = behind = 0
    pad_value = float("{0:.2f}".format(0.))
    top_pad, bottom_pad, left_pad, right_pad = np.array(layer_config['padding']).flatten()
    parameter_mapping = OrderedDict({0: top_pad, 1: bottom_pad, 2: left_pad, 3: right_pad,
                                     4: pad_type, 5: pad_value, 6: per_channel_pad_data_size,
                                     7: front, 8: behind})
    return parameter_mapping


get_zeropadding2d_mapping = partial(get_padding_mapping, pad_type=0)

get_reflectpadding2d_mapping = partial(get_padding_mapping, pad_type=2)


def get_pooling2d_mapping(in_dict, pooling_type, global_pooling = 0):
    #     Pooling  ::  from  C++   enum PoolMethod { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };
    #       0	pooling_type	0
    #       1	kernel_w	0
    #       11	kernel_h	kernel_w
    #       2	stride_w	1
    #       12	stride_h	stride_w
    #       3	pad_left	0
    #       14	pad_right	pad_left
    #       13	pad_top	pad_left
    #       15	pad_bottom	pad_top
    #       4	global_pooling	0
    #       5	pad_mode	0
    layer_config = in_dict['layer'].get_config()
    if global_pooling == 0:
        kernel_h, kernel_w = layer_config['pool_size']
        stride_h, stride_w = layer_config['strides']

        pad_left = pad_right = pad_bottom = pad_top = 0
        pad_mode = 0
        parameter_mapping = OrderedDict({0: pooling_type, 1: kernel_w, 2: stride_w, 3: pad_left,
                                     4: global_pooling, 5: pad_mode,
                                     11: kernel_h, 12: stride_h, 13: pad_top, 14: pad_right, 15: pad_bottom})
    else:
        parameter_mapping = OrderedDict({0: pooling_type, 4: global_pooling})
    return parameter_mapping


get_maxpooling2d_mapping = partial(get_pooling2d_mapping, pooling_type=0, global_pooling=0)

get_averagepooling2d_mapping = partial(get_pooling2d_mapping, pooling_type=1, global_pooling=0)

get_globalmaxpooling2d_mapping = partial(get_pooling2d_mapping, pooling_type=0, global_pooling=1)

get_globalaveragepooling2d_mapping = partial(get_pooling2d_mapping, pooling_type=1, global_pooling=1)


def get_multiply_mapping(in_dict):
    return get_merge_mapping(in_dict, op_type=0)


def get_add_mapping(in_dict):
    return get_merge_mapping(in_dict, op_type=1)


def get_merge_mapping(in_dict, op_type):
    # Add, Multiply, Max ->  C++ class Eltwise : public Layer
    # enum OperationType { Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2 };
    # Eltwise
    # 0  op_type     0
    # 1  coeffs     []
    # TODO :: support other ops
    parameter_mapping = OrderedDict({0: op_type})
    return parameter_mapping


def get_upsampling2d_mapping(in_dict):
    #     Interp
    #     0	resize_type	0	 #FIXED TO 1 (nearest), 2 (bilinear)
    #     1	height_scale	1.f
    #     2	width_scale	1.f
    #     3	output_height	0
    #     4	output_width	0
    layer_config = in_dict['layer'].get_config()
    resize_type = 1
    # print(layer_config)
    if 'interpolation' in layer_config:
        if layer_config['interpolation'] == 'bilinear':
            resize_type = 2

    height_scale, width_scale = layer_config['size']
    parameter_mapping = OrderedDict({0: resize_type, 1: float("{0:.2f}".format(height_scale)),
                                     2: float("{0:.2f}".format(width_scale)),
                                     # TODO :: Clarify lines below
                                     # 3: <>, 4: <>
                                     })
    return parameter_mapping


def get_conv_padding(input_size, output_size, kernel_size, stride_size, dilation_rate):
    t_pad = kernel_size + stride_size * (output_size - 1) - input_size
    t_pad = max(t_pad, 0)
    f_pad = s_pad = 0
    if t_pad > 0:
        f_pad = t_pad // 2
        s_pad = t_pad - f_pad
    return f_pad, s_pad


def get_deconv_padding(input_size, output_size, kernel_size, stride_size, dilation_rate):
    if stride_size != 1:
        # This should be covered by the accurate unit tests
        assert (kernel_size != 1), 'This check this case separately'

    t_pad = (kernel_size - stride_size) + input_size * stride_size - output_size
    t_pad = max(t_pad, 0)
    f_pad = s_pad = 0
    if t_pad > 0:
        f_pad = t_pad // 2
        s_pad = t_pad - f_pad

    return f_pad, s_pad


def get_conv2dtranspose_mapping(in_dict):
    # Deconvolution
    # 0	num_output	0	weight bias
    # 1	kernel_w	0
    # 2	dilation_w	1
    # 3	stride_w	1
    # 4	pad_left	0
    # 5	bias_term	0
    # 6	weight_data_size	0
    # 9	activation_type	0
    # 10	activation_params	[ ]
    # 11	kernel_h	kernel_w
    # 12	dilation_h	dilation_w
    # 13	stride_h	stride_w
    # 15	pad_right	pad_left
    # 14	pad_top	pad_left
    # 16	pad_bottom	pad_top
    # 18	output_pad_right	0
    # 19	output_pad_bottom	output_pad_right
    # 20	output_w	0
    # 21	output_h	output_w
    layer = in_dict['layer']
    layer_config = layer.get_config()
    if layer_config['use_bias']:
        w, b = layer.get_weights()
        w = np.transpose(w, (2, 3, 0, 1))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]
    else:
        assert False, "This branch was not verified"
        w = layer.get_weights()
        in_dict['weight_list'] += [w.flatten()]

    num_output = layer_config['filters']
    kernel_h, kernel_w = layer_config['kernel_size']
    dilation_h, dilation_w = layer_config['dilation_rate']
    stride_h, stride_w = layer_config['strides']
    bias_term = int(layer_config['use_bias'])
    weight_data_size = np.prod(w.shape)
    int8_scale_term = 0

    layer_input_shape = get_valid_shape(layer.input_shape)
    layer_output_shape = get_valid_shape(layer.output_shape)

    assert layer_config['activation'] in activation_type_dict
    activation_type = activation_type_dict[layer_config['activation']]
    # "TODO :: Support https://github.com/Tencent/ncnn/blob/master/tools/ncnnoptimize.cpp"
    assert len(layer_input_shape) == len(layer_output_shape) == 1
    assert dilation_h == dilation_w == 1

    input_y_size, input_x_size = layer_input_shape[0][1:3]
    output_y_size, output_x_size = layer_output_shape[0][1:3]
    if layer_config['padding'] == 'same':
        pad_left, pad_right = get_deconv_padding(input_x_size, output_x_size, kernel_w, stride_w, dilation_w)
        pad_top, pad_bottom = get_deconv_padding(input_y_size, output_y_size, kernel_h, stride_h, dilation_h)
    else:
        assert True, 'Check This Code Branch'
        pad_left = pad_right = pad_top = pad_bottom = 0

    # output_h, output_w = output_y_size, output_x_size

    parameter_mapping = OrderedDict({0: num_output, 1: kernel_w, 2: dilation_w, 3: stride_w, 4: pad_left,
                                     5: bias_term, 6: weight_data_size, 8: int8_scale_term, 9: activation_type,
                                     # TODO :: Check Acrivation params 10:,

                                     11: kernel_h, 12: dilation_h, 13: stride_h, 14: pad_top, 15: pad_right,
                                     16: pad_bottom,
                                     # TODO :: FILL This params 18:<>, 19:<>,
                                     # 20: output_w, 21: output_h
                                     })
    return parameter_mapping


def get_dense_mapping(in_dict):
    # InnerProduct	0	num_output	        0	weight bias
    #               1	bias_term	        0
    #               2	weight_data_size	0
    #               8	int8_scale_term	    0
    #               9	activation_type	    0
    #               10	activation_params	[]

    layer = in_dict['layer']
    layer_config = layer.get_config()
    if layer_config['use_bias']:
        w, b = layer.get_weights()
        w = np.transpose(w, (1, 0))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]
    else:
        # TODO :: Try to skip bias add, currently zero bias added
        w, = layer.get_weights()
        c_size, f_size = w.shape
        w = np.transpose(w, (1, 0))
        b = np.zeros((f_size,))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]

    num_output = layer_config['units']
    # TODO :: check use bias = False
    bias_term = int(True)  # int(layer_config['use_bias'])
    weight_data_size = np.prod(w.shape)

    assert layer_config['activation'] in activation_type_dict
    activation_type = activation_type_dict[layer_config['activation']]

    parameter_mapping = OrderedDict({0: num_output, 1: bias_term, 2: weight_data_size, 9: activation_type})
    return parameter_mapping


def get_conv2d_mapping(in_dict):
    #     Convolution
    #     0	    num_output	        0	weight bias
    #     1	    kernel_w	        0
    #     2	    dilation_w	        1
    #     3	    stride_w	        1
    #     4	    pad_left	        0
    #     5	    bias_term	        0
    #     6	    weight_data_size	0
    #     8	    int8_scale_term	    0
    #     9	    activation_type	    0
    #     10	activation_params	[ ]
    #     11	kernel_h	kernel_w
    #     12	dilation_h	dilation_w
    #     13	stride_h	stride_w
    #     15	pad_right	pad_left
    #     14	pad_top	pad_left
    #     16	pad_bottom	pad_top
    #     17	impl_type	0
    #     18	pad_value	0.f

    layer = in_dict['layer']
    layer_config = layer.get_config()
    if layer_config['use_bias']:
        w, b = layer.get_weights()
        w = np.transpose(w, (3, 2, 0, 1))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]
    else:
        # TODO :: Try to skip bias add, currently zero bias added
        w, = layer.get_weights()
        _, _, c_size, f_size = w.shape
        w = np.transpose(w, (3, 2, 0, 1))
        b = np.zeros((f_size,))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]

    num_output = layer_config['filters']
    kernel_h, kernel_w = layer_config['kernel_size']
    dilation_h, dilation_w = layer_config['dilation_rate']
    stride_h, stride_w = layer_config['strides']
    # TODO :: check use bias = False
    bias_term = int(True)  # int(layer_config['use_bias'])
    weight_data_size = np.prod(w.shape)
    int8_scale_term = 0

    layer_input_shape = get_valid_shape(layer.input_shape)
    layer_output_shape = get_valid_shape(layer.output_shape)

    assert layer_config['activation'] in activation_type_dict
    activation_type = activation_type_dict[layer_config['activation']]
    # "TODO :: Support https://github.com/Tencent/ncnn/blob/master/tools/ncnnoptimize.cpp"
    assert len(layer_input_shape) == len(layer_output_shape) == 1
    assert dilation_h == dilation_w == 1

    if layer_config['padding'] == 'same':
        input_y_size, input_x_size = layer_input_shape[0][1:3]
        output_y_size, output_x_size = layer_output_shape[0][1:3]
        pad_left, pad_right = get_conv_padding(input_x_size, output_x_size, kernel_w, stride_w, dilation_w)
        pad_top, pad_bottom = get_conv_padding(input_y_size, output_y_size, kernel_h, stride_h, dilation_h)
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0

    parameter_mapping = OrderedDict({0: num_output, 1: kernel_w, 2: dilation_w, 3: stride_w, 4: pad_left,
                                     5: bias_term, 6: weight_data_size, 8: int8_scale_term, 9: activation_type,
                                     # TODO :: FILL Acrivation params 10:,
                                     11: kernel_h, 12: dilation_w, 13: stride_h, 14: pad_top, 15: pad_right,
                                     16: pad_bottom
                                     # TODO :: FILL This params 17:<>, 18:<>,
                                     })
    return parameter_mapping


def get_depthwiseconv2d_mapping(in_dict):
    #     ConvolutionDepthWise
    #     0	num_output	0	weight bias
    #     1	kernel_w	0
    #     2	dilation_w	1
    #     3	stride_w	1
    #     4	pad_left	0
    #     5	bias_term	0
    #     6	weight_data_size	0
    #     8	int8_scale_term	0
    #     9	activation_type	0
    #     10	activation_params	[ ]
    #     11	kernel_h	kernel_w
    #     12	dilation_h	dilation_w
    #     13	stride_h	stride_w
    #     15	pad_right	pad_left
    #     14	pad_top	pad_left
    #     16	pad_bottom	pad_top
    #     17	impl_type	0
    #     18	pad_value	0.f

    layer = in_dict['layer']
    layer_config = layer.get_config()
    if layer_config['use_bias']:
        w, b = layer.get_weights()
        w = np.transpose(w, (3, 2, 0, 1))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]
    else:
        # TODO :: Try to skip bias add, currently zero bias added
        w, = layer.get_weights()
        _, _, c_size, f_size = w.shape
        w = np.transpose(w, (3, 2, 0, 1))
        b = np.zeros((c_size,))
        in_dict['weight_list'] += [w.flatten(), b.flatten()]

    kernel_h, kernel_w = layer_config['kernel_size']
    dilation_h, dilation_w = layer_config['dilation_rate']
    stride_h, stride_w = layer_config['strides']
    # TODO Check use_bias=False
    bias_term = int(True)  # int(layer_config['use_bias'])
    weight_data_size = np.prod(w.shape)
    int8_scale_term = 0

    layer_input_shape = get_valid_shape(layer.input_shape)
    layer_output_shape = get_valid_shape(layer.output_shape)

    num_output = layer_output_shape[0][-1]
    group = num_output

    assert layer_config['activation'] in activation_type_dict
    activation_type = activation_type_dict[layer_config['activation']]
    # "TODO :: Support https://github.com/Tencent/ncnn/blob/master/tools/ncnnoptimize.cpp"
    assert len(layer_input_shape) == len(layer_output_shape) == 1
    assert dilation_h == dilation_w == 1

    if layer_config['padding'] == 'same':
        input_y_size, input_x_size = layer_input_shape[0][1:3]
        output_y_size, output_x_size = layer_output_shape[0][1:3]
        pad_left, pad_right = get_conv_padding(input_x_size, output_x_size, kernel_w, stride_w, dilation_w)
        pad_top, pad_bottom = get_conv_padding(input_y_size, output_y_size, kernel_h, stride_h, dilation_h)
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0

    parameter_mapping = OrderedDict({0: num_output, 1: kernel_w, 2: dilation_w, 3: stride_w, 4: pad_left,
                                     5: bias_term, 6: weight_data_size, 7: group,
                                     8: int8_scale_term, 9: activation_type,
                                     # TODO :: FILL Acrivation params 10:,
                                     11: kernel_h, 12: dilation_w, 13: stride_h, 14: pad_top, 15: pad_right,
                                     16: pad_bottom
                                     # TODO :: FILL This params 17:<>, 18:<>,
                                     })
    return parameter_mapping


def extract_normalization_weights(layer):
    layer_config = layer.get_config()

    if layer_config['scale'] and layer_config['center']:
        scale, beta, moving_mean, moving_variance = layer.get_weights()
    elif (not layer_config['scale']) and layer_config['center']:
        beta, moving_mean, moving_variance = layer.get_weights()
        scale = 1. + 0 * moving_mean.flatten()
    elif layer_config['scale'] and (not layer_config['center']):
        scale, moving_mean, moving_variance = layer.get_weights()
        beta = 0 * moving_mean.flatten()
    elif (not layer_config['scale']) and (not layer_config['center']):
        scale, moving_mean, moving_variance = layer.get_weights()
        scale = 1. + 0 * moving_mean.flatten()
        beta = 0 * moving_mean.flatten()
    else:
        assert False, "This branch was not verified"

    return scale, beta, moving_mean, moving_variance


def get_batchnormalization_mapping(in_dict):
    # BatchNorm   0   channels    0   slope   mean    variance    bias
    #             1   eps 0.f

    layer = in_dict['layer']
    scale, beta, moving_mean, moving_variance = extract_normalization_weights(layer)
    in_dict['weight_list'] += [scale.flatten(), moving_mean.flatten(),
                               moving_variance.flatten(), beta.flatten()]

    layer_output_shape = get_valid_shape(layer.output_shape)
    num_output = layer_output_shape[0][-1]
    epsilon = layer.get_config()['epsilon']

    parameter_mapping = OrderedDict({0: num_output, 1: epsilon})
    return parameter_mapping


def get_instancenormalization_mapping(in_dict):
    # InstanceNorm      0   channels    0   gamma bias
    #                   1   eps 0.f

    layer = in_dict['layer']
    scale, beta, moving_mean, moving_variance = extract_normalization_weights(layer)
    in_dict['weight_list'] += [scale.flatten(), beta.flatten()]

    layer_output_shape = get_valid_shape(layer.output_shape)
    num_output = layer_output_shape[0][-1]
    epsilon = layer.get_config()['epsilon']

    parameter_mapping = OrderedDict({0: num_output, 1: epsilon})
    return parameter_mapping


def get_tanh_mapping(in_dict):
    return OrderedDict({})


def get_clip_mapping(in_dict):
    # Clip	0	min	-FLT_MAX
    #       1	max	FLT_MAX
    layer = in_dict['layer']
    layer_config = layer.get_config()
    parameter_mapping = OrderedDict({})
    parameter_mapping[0] = float("{0:.7f}".format(float(layer_config['min_value'])))
    parameter_mapping[1] = float("{0:.7f}".format(float(layer_config['max_value'])))
    return parameter_mapping


def get_relu_mapping(in_dict):
    # ReLU	0	slope	0.f
    layer = in_dict['layer']
    layer_config = layer.get_config()
    slope = 0
    parameter_mapping = OrderedDict({})
    if 'slope' in layer_config:
        slope = layer_config['slope']
    slope = float("{0:.7f}".format(float(slope)))
    parameter_mapping[0] = slope

    if 'max_value' in layer_config:
        max_value = layer_config['max_value']
        if max_value is not None:
            assert True, 'This branch should not be executed!'
            max_value = float("{0:.7f}".format(float(max_value)))
            parameter_mapping[1] = max_value
    return parameter_mapping


def get_leakyrelu_mapping(in_dict):
    # ReLU	0	slope	0.f
    layer = in_dict['layer']
    layer_config = layer.get_config()
    slope = float("{0:.7f}".format(layer_config['alpha']))
    parameter_mapping = OrderedDict({0: slope})
    return parameter_mapping


def get_softmax_mapping(in_dict):
    # Softmax	0	axis	0
    # Fix BUG
    fix_bug = 1
    # NCNN
    layer_config = in_dict['layer'].get_config()
    if 'axis' in layer_config:
        axis = layer_config['axis']
    else:
        axis = -1

    axis = fix_axis_value(in_dict, axis)
    parameter_mapping = OrderedDict({0: int(axis), 1: fix_bug})
    return parameter_mapping


def get_sigmoid_mapping(in_dict):
    # Sigmoid	No params
    parameter_mapping = OrderedDict({})
    return parameter_mapping


def get_activation_mapping(in_dict):
    # Activation alias mapping
    layer_config = in_dict['layer'].get_config()
    mapping_function_name = '_'.join(['get', str(layer_config['activation']).lower(), 'mapping'])
    mapping_function = globals()[mapping_function_name]
    parameter_mapping = mapping_function(in_dict)
    return parameter_mapping


def get_concatenate_mapping(in_dict):
    # Concat	0	axis	0
    layer_config = in_dict['layer'].get_config()
    axis = layer_config['axis']
    axis = fix_axis_value(in_dict, axis)
    parameter_mapping = OrderedDict({0: axis})
    return parameter_mapping


def get_parameter_string(in_dict, mapping_function_name):
    mapping_function = globals()[mapping_function_name]
    parameter_mapping = mapping_function(in_dict)
    parameter_string = ' '.join([f'{key}={value}' for key, value in parameter_mapping.items()])
    return parameter_string


def get_model_string(model, magic_number, blob_set, string_list):
    layer_count, blob_count = blob_set
    string_list = [str(magic_number), f'{layer_count} {blob_count}']
    return string_list


def get_layer_string(in_dict, export_shapes=False):
    batch_size = in_dict['batch_size']
    layer = in_dict['layer']
    layer_type, mapping_function_name = get_layer_type(layer)
    layer_name = get_layer_name(layer)

    in_out_string, split_string, tensor_names = get_in_out_string(in_dict)
    blob_shape_string = get_blob_shape_string(layer, batch_size)
    parameter_string = get_parameter_string(in_dict, mapping_function_name)
    array_key = str(in_dict['array_key'])

    # TODO :: autodetect line length
    max_line_length = 36
    assert len(layer_type) < max_line_length
    assert len(layer_name) < max_line_length

    string_list = []

    if export_shapes:
        string_list.append(
            f'{layer_type: <36}{layer_name: <36}{in_out_string} {array_key}={blob_shape_string} {parameter_string}')
    else:
        string_list.append(
            f'{layer_type: <36}{layer_name: <36}{in_out_string} {parameter_string}')
    if split_string is not None:
        string_list.append(split_string)
    return string_list, layer_name, tensor_names


def conver_model(model, debug=True, export_shapes=False):
    # magic number: 7767517
    magic_number = 7767517
    # TODO :: clarify this
    array_key = '-23330'
    batch_size = 1
    weight_list = []

    split_info = {}

    if debug:
        print("Export graph and weights")
    string_list = []
    layer_name_list = []
    tensor_name_list = []

    outbound_dict, index_dict = get_outbound_nodes(model.get_config())

    for layer in tqdm(model.layers):
        config_dict = {'layer': layer, 'array_key': array_key, 'batch_size': batch_size,
                       'weight_list': [], 'model_output_names': model.output_names,
                       'split_info': split_info, 'outbound_dict': outbound_dict}
        add_string, layer_name, tensor_names = get_layer_string(config_dict, export_shapes)
        weight_list += [[layer.__class__.__name__, config_dict['weight_list']]]
        string_list += add_string
        layer_name_list += [layer_name]
        tensor_name_list += tensor_names

    tmp_list = []
    for item in split_info:
        for tmp in split_info[item]:
            tmp_list.append(split_info[item][tmp])

    blob_set = (len(model.layers), len(set(tensor_name_list)))

    string_list = get_model_string(model, magic_number, blob_set=blob_set, string_list=string_list) + string_list

    return string_list, weight_list, layer_name_list
