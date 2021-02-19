from copy import deepcopy
from optimization.graph.layer_template import Conv2D_config_template, Softmax_config_template

info_Conv2DSoftmax = 'Conv2DSoftmax: Conv2D with Softmax activation can not be transformed. Transforming this to ' \
                       'pair Conv2D->Softmax'

def transfer_Conv2DSoftmax_Conv2D(src_model, dst_model, transfer_rule):
    dst_model.get_layer(transfer_rule['dst']).set_weights(src_model.get_layer(transfer_rule['src']).get_weights())


def transfer_Conv2DSoftmax_Softmax(src_model, dst_model, transfer_rule):
    pass


def detect_transform_Conv2DSoftmax(keras_config):
    index_list = []
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] == 'Conv2D':
            if item['config']['activation'] == 'softmax':
                index_list.append(i)
    return index_list

def check_Conv2DSoftmax_transfrom(keras_config):
    return len(detect_transform_Conv2DSoftmax(keras_config)) > 0

def apply_transform_Conv2DSoftmax(keras_config):
    index_list = detect_transform_Conv2DSoftmax(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        # Transfer DepthWise
        i_layer_config = deepcopy(Conv2D_config_template)
        # TODO :: check unique name
        prev_name = i_layer_config['name'] = r_layer_config['name'] + f'_act_{i}'
        for key in Conv2D_config_template['config'].keys():
            if key in r_layer_config['config']:
                i_layer_config['config'][key] = r_layer_config['config'][key]
        i_layer_config['inbound_nodes'] = r_layer_config['inbound_nodes']
        i_layer_config['config']['name'] = i_layer_config['name']
        i_layer_config['config']['activation'] = 'linear'
        keras_config['layers'].insert(i, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {'transfer_call': transfer_Conv2DSoftmax_Conv2D,
                                                             'src': r_layer_config['name'],
                                                             'dst': i_layer_config['name']}

        # Transfer Conv
        i_layer_config = deepcopy(Softmax_config_template)
        for key in set(Softmax_config_template['config'].keys()):
            if key in r_layer_config['config']:
                i_layer_config['config'][key] = r_layer_config['config'][key]
        i_layer_config['name'] = r_layer_config['name']
        i_layer_config['inbound_nodes'] = [[[prev_name, 0, 0, {}]]]

        keras_config['layers'].insert(i + 1, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {'transfer_call': transfer_Conv2DSoftmax_Softmax,
                                                             'src': r_layer_config['name'],
                                                             'dst': i_layer_config['name']}

        index_list = detect_transform_Conv2DSoftmax(keras_config)
    return keras_config, weight_transfer_rule_dict

