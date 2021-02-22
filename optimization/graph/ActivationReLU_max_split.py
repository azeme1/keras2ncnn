from copy import deepcopy
from optimization.graph.layer_template import ReLU_config_template, Clip_config_template

info_ActivationReLU_max = 'Activation(relu6): Activation(relu6) is not supported withing NCNN, split by ordinal ReLU and Clip'

def transfer_ActivationReLU_max(src_model, dst_model, transfer_rule):
    pass

def detect_transform_ActivationReLU_max(keras_config):
    index_list = []
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] == 'Activation':
            if item['config']['activation'] == 'relu6':
                index_list.append(i)
    return index_list

def check_ActivationReLU_max_transfrom(keras_config):
    return len(detect_transform_ActivationReLU_max(keras_config)) > 0

def apply_transform_ActivationReLU_max(keras_config):
    index_list = detect_transform_ActivationReLU_max(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        # Transfer DepthWise
        i_layer_config = deepcopy(ReLU_config_template)
        # TODO :: check unique name
        prev_name = i_layer_config['name'] = r_layer_config['name'] + f'_dwc_{i}'
        for key in i_layer_config['config'].keys():
            if key in r_layer_config['config']:
                i_layer_config['config'][key] = r_layer_config['config'][key]
        # Copy parameters
        i_layer_config['config']['threshold'] = 0
        i_layer_config['config']['max_value'] = None
        i_layer_config['inbound_nodes'] = r_layer_config['inbound_nodes']
        i_layer_config['config']['name'] = i_layer_config['name']
        keras_config['layers'].insert(i, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {'transfer_call': transfer_ActivationReLU_max,
                                                             'src': r_layer_config['name'],
                                                             'dst': i_layer_config['name']}

        # Transfer Clip parameters
        i_layer_config = deepcopy(Clip_config_template)
        for key in i_layer_config['config'].keys():
            if key in r_layer_config['config']:
                i_layer_config['config'][key] = r_layer_config['config'][key]
        i_layer_config['config']['max_value'] = 6
        i_layer_config['config']['min_value'] = 0
        i_layer_config['name'] = r_layer_config['name']
        i_layer_config['inbound_nodes'] = [[[prev_name, 0, 0, {}]]]
        keras_config['layers'].insert(i + 1, i_layer_config)
        weight_transfer_rule_dict[i_layer_config['name']] = {'transfer_call': transfer_ActivationReLU_max,
                                                             'src': r_layer_config['name'],
                                                             'dst': i_layer_config['name']}

        index_list = detect_transform_ActivationReLU_max(keras_config)
    return keras_config, weight_transfer_rule_dict

