import numpy as np
from converter.model_adaptation import rename_layer


info_Conv2DBatchNormalization = 'Conv2D->BatchNormalization: Subsequent linear operations can be easily merged for inference'


def transfer_Conv2DBatchNormalization_Conv2D(src_model, dst_model, transfer_rule):
    layer_c = src_model.get_layer(transfer_rule['src_c'])
    weigths_c = layer_c.get_weights()
    layer_b = src_model.get_layer(transfer_rule['src_b'])
    weigths_b = layer_b.get_weights()

    eps = layer_b.get_config()['epsilon']
    weight, bias = weigths_c
    gamma, beta, mean, var = weigths_b

    a = gamma / np.sqrt(var + eps)
    weight = weight * a
    bias = a * (bias - mean) + beta

    dst_model.get_layer(transfer_rule['dst']).set_weights([weight, bias])


def get_outbound_nodes(keras_config):
    outbound_dict = {}
    index_dict = {}
    for _i, _layer in enumerate(keras_config['layers']):
        out_node_name = _layer['name']
        index_dict[out_node_name] = _i
        if len(_layer['inbound_nodes']) == 0:
            continue
        in_node_name = _layer['inbound_nodes'][0][0][0]
        if in_node_name in outbound_dict:
            outbound_dict[in_node_name] += [out_node_name]
        else:
            outbound_dict[in_node_name] = [out_node_name]

    return outbound_dict, index_dict


def detect_transform_Conv2DBatchNormalization(keras_config):
    index_list = []
    outbound_dict, index_dict = get_outbound_nodes(keras_config)
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] == 'BatchNormalization':
            in_node_name = item['inbound_nodes'][0][0][0]
            in_node_class_name = keras_config['layers'][index_dict[in_node_name]]['class_name']
            if in_node_class_name == 'Conv2D':
                if len(outbound_dict[in_node_name]) == 1:
                    index_list.append(i)
    return index_list


def check_Conv2DBatchNormalization(keras_config):
    return len(detect_transform_Conv2DBatchNormalization(keras_config)) > 0


def apply_transform_Conv2DBatchNormalization(keras_config):
    index_list = detect_transform_Conv2DBatchNormalization(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        src_name = r_layer_config['name']
        dst_name = r_layer_config['inbound_nodes'][0][0][0]
        keras_config = rename_layer(keras_config, src_name, dst_name)
        merged_dst = dst_name + '_M'
        keras_config = rename_layer(keras_config, dst_name, merged_dst)
        weight_transfer_rule_dict[merged_dst] = {'transfer_call': transfer_Conv2DBatchNormalization_Conv2D,
                                                 'src_c': dst_name,
                                                 'src_b': src_name,
                                                 'dst': merged_dst}
        index_list = detect_transform_Conv2DBatchNormalization(keras_config)
    return keras_config, weight_transfer_rule_dict