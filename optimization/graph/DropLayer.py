import numpy as np
from copy import deepcopy

from converter.model_adaptation import get_outbound_nodes, rename_layer

info_DropLayer = 'Dropout: Some nodes can de skipped in inference mode'


def transfer_DropLayer(src_model, dst_model, transfer_rule):
    pass


def detect_transform_DropLayer(keras_config):
    index_list = []
    for i, item in enumerate(keras_config['layers']):
        if item['class_name'] in ['Dropout', 'DropConnect']:
            index_list.append(i)
    return index_list


def check_DropLayer(keras_config):
    return len(detect_transform_DropLayer(keras_config)) > 0


def apply_transform_DropLayer(keras_config):
    index_list = detect_transform_DropLayer(keras_config)
    weight_transfer_rule_dict = {}
    while len(index_list) > 0:
        i = index_list[0]
        r_layer_config = keras_config['layers'].pop(i)
        src_name = r_layer_config['name']
        dst_name = r_layer_config['inbound_nodes'][0][0][0]
        keras_config = rename_layer(keras_config, src_name, dst_name)

        index_list = detect_transform_DropLayer(keras_config)
    return keras_config, weight_transfer_rule_dict
