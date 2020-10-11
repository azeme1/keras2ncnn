from copy import deepcopy
from extra_layers.OutputSplit import OutputSplit
from tensorflow.keras.models import Model

def is_multi_output(keras_model_config):
    for layer_index, layer_item in enumerate(keras_model_config['layers']):
        inbound_nodes = layer_item['inbound_nodes']
        if len(inbound_nodes) == 0:
            continue
        else:
            inbound_node_list = [node_item[0] for node_item in inbound_nodes[0]]
            if len(inbound_node_list) > 1:
                return True
    return False

def rename_layer(model_config, src_name, dst_name):
    for layer_item in model_config['input_layers']:
        input_condition = layer_item[0] == src_name
        if input_condition:
            # print('Rename Input:: ', layer_item[0])
            layer_item[0] = dst_name

    for layer_item in model_config['output_layers']:
        output_condition = layer_item[0] == src_name
        if output_condition:
            # print('Rename Output:: ', layer_item[0])
            layer_item[0] = dst_name
    #     print(rename_condition,  layer_list[0])

    for layer_item in model_config['layers']:
        name_condition = layer_item['name'] == src_name
        if name_condition:
            # print('Rename Layers:: name', layer_item['name'])
            layer_item['name'] = dst_name

        name_condition = layer_item['config']['name'] == src_name
        if name_condition:
            # print('Rename Layers Config:: config', layer_item['config']['name'])
            layer_item['config']['name'] = dst_name
            #             print('               Rename Layers Config:: config', layer_item['config']['name'])

        if len(layer_item['inbound_nodes']) > 0:
            for item in layer_item['inbound_nodes'][0]:
                inbound_condition = item[0] == src_name
                if inbound_condition:
                    # print('Rename Layers Inbound::', item[0])
                    item[0] = dst_name
    return model_config

def get_outbound_nodes(keras_model_config):
    outbound_nodes_dict = {}
    layer_index_dict = {}

    for layer_index, layer_item in enumerate(keras_model_config['layers']):
        out_node_name = layer_item['name']
        layer_index_dict[out_node_name] = layer_index

        inbound_nodes = layer_item['inbound_nodes']
        if len(inbound_nodes) == 0:
            continue
        else:
            inbound_node_list = [node_item[0] for node_item in inbound_nodes[0]]

        for in_node_name in inbound_node_list:
            # print(in_node_name, out_node_name)
            if in_node_name in outbound_nodes_dict:
                outbound_nodes_dict[in_node_name].append(out_node_name)
            else:
                outbound_nodes_dict[in_node_name] = [out_node_name]

    for in_node_name in keras_model_config['output_layers']:
        _in_node_name = in_node_name[0]
        outbound_nodes_dict[_in_node_name] = []

    return outbound_nodes_dict, layer_index_dict

def split_output_nodes(keras_model_config):
    protected_class_names = ['OutputSplit']

    nccn2keras_layer_list = []
    count_index = 0
    model_config_layers = deepcopy(keras_model_config['layers'])
    model_config = deepcopy(keras_model_config)

    outbound_nodes_dict, layer_index_dict = get_outbound_nodes(keras_model_config)

    for layer_item in model_config_layers:
        out_node_name = layer_item['name']
        nccn2keras_layer_list.append(layer_item)
        split_condition = len(outbound_nodes_dict[out_node_name]) > 1 and (
            not (layer_item['class_name'] in protected_class_names))
        # print(split_condition)
        if split_condition:
            count_index += 1
            # print("-----", out_node_name)
            # print(layer_item)
            # print(outbound_nodes_dict[out_node_name])
            split_node_name = f'ncnn_split_{str(count_index)}'
            split_layer_config = {'class_name': 'OutputSplit', 'config': {'name': split_node_name,
                                                                          'trainable': True,
                                                                          'dtype': 'float32',
                                                                          'count': len(
                                                                              outbound_nodes_dict[out_node_name])},
                                  'name': split_node_name,
                                  'inbound_nodes': [[[out_node_name, 0, 0, {}]]]}
            nccn2keras_layer_list.append(split_layer_config)
            use_index = 0
            for item in outbound_nodes_dict[out_node_name]:
                p_item = model_config_layers[layer_index_dict[item]]['inbound_nodes']
                # print('----')
                for t_item in p_item[0]:
                    if t_item[0] == out_node_name:
                        # print('                  ', t_item)
                        t_item[0] = split_node_name
                        t_item[2] = use_index
                        use_index += 1
                        # print('                  ', t_item)
        else:
            continue

    model_config['layers'] = nccn2keras_layer_list
    return model_config

def adapt_keras_model(keras_model, model_name):
    keras_model_config = keras_model.get_config()
    keras_model_config['name'] = model_name
    assert len(keras_model_config['input_layers']) == len(keras_model_config['output_layers']) == 1, 'Multi IN/OUT is not supported'
    if is_multi_output(keras_model_config):
        adapted_model_config = split_output_nodes(keras_model_config)
        adapted_model_config = rename_layer(adapted_model_config, keras_model_config['input_layers'][0][0], 'data')
        adapted_model_config = rename_layer(adapted_model_config, keras_model_config['output_layers'][0][0], 'output')
        adapted_keras_model = Model.from_config(adapted_model_config, custom_objects={'OutputSplit': OutputSplit})
    else:
        adapted_keras_model = Model.from_config(keras_model_config)
    adapted_keras_model.set_weights(keras_model.get_weights())
    return adapted_keras_model

