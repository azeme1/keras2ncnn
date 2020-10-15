import numpy as np
from copy import deepcopy

from optimization.graph.layer_template import DepthwiseConv2D_config_template

info_BatchNormalization_DepthwiseConv2D = 'BatchNormalization->DepthwiseConv2D: Single BatchNormalization is ' \
                                          'recommended to transform in to DepthwiseConv2D for subsequent Activation ' \
                                          'merge...'


def transfer_BatchNormalization_DepthwiseConv2D(src_model, dst_model, transfer_rule):
    gamma, beta, mean, var = src_model.get_layer(transfer_rule['src']).get_weights()
    eps = src_model.get_layer(transfer_rule['src']).get_config()['epsilon']
    a = gamma / np.sqrt(var + eps)
    weight = a.reshape((1, 1, -1, 1))
    bias = -a * mean + beta

    dst_model.get_layer(transfer_rule['dst']).set_weights([weight, bias])


def detect_transform_BatchNormalization_DepthwiseConv2D(keras_config):
    index_list = []
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] == 'BatchNormalization':
            index_list.append(i)
    return index_list


def check_BatchNormalization_DepthwiseConv2D(keras_config):
    return len(detect_transform_BatchNormalization_DepthwiseConv2D(keras_config)) > 0


def apply_transform_BatchNormalization_DepthwiseConv2D(keras_config):
    index_list = detect_transform_BatchNormalization_DepthwiseConv2D(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        i_layer_config = deepcopy(DepthwiseConv2D_config_template)

        i_layer_config['inbound_nodes'] = r_layer_config['inbound_nodes']
        i_layer_config['config']['name'] = i_layer_config['name'] = r_layer_config['name']
        i_layer_config['config']['use_bias'] = True
        i_layer_config['config']['kernel_size'] = (1, 1)
        keras_config['layers'].insert(i, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {
            'transfer_call': transfer_BatchNormalization_DepthwiseConv2D,
            'src': r_layer_config['name'],
            'dst': i_layer_config['name']}
        index_list = detect_transform_BatchNormalization_DepthwiseConv2D(keras_config)
    return keras_config, weight_transfer_rule_dict
