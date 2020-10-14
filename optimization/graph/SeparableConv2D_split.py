from copy import deepcopy
from optimization.graph.layer_template import Conv2D_config_template, DepthwiseConv2D_config_template

info_SeparableConv2D = 'SeparableConv2D: It looks like was not implemented in NCNN, splitting it to DepthwiseConv2D ' \
                       'and Conv2D '

def transfer_SeparableConv2D_DepthwiseConv2D(src_model, dst_model, transfer_rule):
    _weigths = src_model.get_layer(transfer_rule['src']).get_weights()
    dst_model.get_layer(transfer_rule['dst']).set_weights([_weigths[0]])

def transfer_SeparableConv2D_Conv2D(src_model, dst_model, transfer_rule):
    _weigths = src_model.get_layer(transfer_rule['src']).get_weights()
    dst_model.get_layer(transfer_rule['dst']).set_weights(_weigths[1:])

def detect_transform_SeparableConv2D(keras_config):
    index_list = []
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] == 'SeparableConv2D':
            index_list.append(i)
    return index_list

def check_SeparableConv2D_transfrom(keras_config):
    return len(detect_transform_SeparableConv2D(keras_config)) > 0

def apply_transform_SeparableConv2D(keras_config):
    index_list = detect_transform_SeparableConv2D(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        # Transfer DepthWise
        i_layer_config = deepcopy(DepthwiseConv2D_config_template)
        # TODO :: check unique name
        prev_name = i_layer_config['name'] = r_layer_config['name'] + f'_dwc_{i}'
        for key in DepthwiseConv2D_config_template['config'].keys():
            if key in r_layer_config['config']:
                i_layer_config['config'][key] = r_layer_config['config'][key]
        i_layer_config['inbound_nodes'] = r_layer_config['inbound_nodes']
        i_layer_config['config']['name'] = i_layer_config['name']
        i_layer_config['config']['use_bias'] = False
        keras_config['layers'].insert(i, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {'transfer_call': transfer_SeparableConv2D_DepthwiseConv2D,
                                                             'src': r_layer_config['name'],
                                                             'dst': i_layer_config['name']}

        # Transfer Conv
        i_layer_config = deepcopy(Conv2D_config_template)
        for key in set(Conv2D_config_template['config'].keys()) - {'kernel_size', 'strides', 'dilation_rate'}:
            if key in r_layer_config['config']:
                i_layer_config['config'][key] = r_layer_config['config'][key]
        i_layer_config['name'] = r_layer_config['name']
        i_layer_config['inbound_nodes'] = [[[prev_name, 0, 0, {}]]]

        keras_config['layers'].insert(i + 1, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {'transfer_call': transfer_SeparableConv2D_Conv2D,
                                                             'src': r_layer_config['name'],
                                                             'dst': i_layer_config['name']}

        index_list = detect_transform_SeparableConv2D(keras_config)
    return keras_config, weight_transfer_rule_dict

