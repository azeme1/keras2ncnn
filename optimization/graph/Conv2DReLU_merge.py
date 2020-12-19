import numpy as np
from tensorflow.keras.layers import ReLU
from converter.model_adaptation import rename_layer, get_outbound_nodes

info_Conv2DReLU = 'Conv2D->ReLU: Inline operations should be merged'


def transfer_Conv2DReLU_Conv2D(src_model, dst_model, transfer_rule):
    dst_model.get_layer(transfer_rule['dst']).set_weights(src_model.get_layer(transfer_rule['src']).get_weights())


# def get_outbound_nodes(keras_config):
#     outbound_dict = {}
#     index_dict = {}
#     for _i, _layer in enumerate(keras_config['layers']):
#         out_node_name = _layer['name']
#         index_dict[out_node_name] = _i
#         if len(_layer['inbound_nodes']) == 0:
#             continue
#         in_node_name = _layer['inbound_nodes'][0][0][0]
#         if in_node_name in outbound_dict:
#             outbound_dict[in_node_name] += [out_node_name]
#         else:
#             outbound_dict[in_node_name] = [out_node_name]
#
#     return outbound_dict, index_dict


def detect_transform_Conv2DReLU(keras_config):
    index_list = []
    outbound_dict, index_dict = get_outbound_nodes(keras_config)
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] == 'ReLU':
            if 'max_value' in item['config']:
                if 'max_value' is not None:
                    continue
            if 'negative_slope' in item['config']:
                if 'negative_slope' > 0.:
                    continue
            in_node_name = item['inbound_nodes'][0][0][0]
            in_node_class_name = keras_config['layers'][index_dict[in_node_name]]['class_name']
            if in_node_class_name in ['Conv2D', 'DepthwiseConv2D']:
                _activation = keras_config['layers'][index_dict[in_node_name]]['config']['activation']
                if _activation in ['linear', 'relu']:
                    if len(outbound_dict[in_node_name]) == 1:
                        index_list.append(i)
    return index_list, index_dict


def check_Conv2DReLU(keras_config):
    return len(detect_transform_Conv2DReLU(keras_config)[0]) > 0


def apply_transform_Conv2DReLU(keras_config):
    index_list, index_dict = detect_transform_Conv2DReLU(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        src_name = r_layer_config['name']
        dst_name = r_layer_config['inbound_nodes'][0][0][0]
        keras_config['layers'][index_dict[dst_name]]['config']['activation'] = 'relu'
        transfer_call = transfer_Conv2DReLU_Conv2D

        keras_config = rename_layer(keras_config, src_name, dst_name)
        merged_dst = dst_name       # + '_M' TODO Decide to rename the layer
        keras_config = rename_layer(keras_config, dst_name, merged_dst)
        weight_transfer_rule_dict[merged_dst] = {'transfer_call': transfer_call,
                                                 'src': dst_name,
                                                 'dst': merged_dst}
        index_list, index_dict = detect_transform_Conv2DReLU(keras_config)
    return keras_config, weight_transfer_rule_dict
