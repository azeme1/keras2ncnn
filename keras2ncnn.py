import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from unit_test.helper import save_config, get_test_item_mean_std
from converter.converter_candidate import conver_model
from converter.model_adaptation import adapt_keras_model
from optimization.optimize_graph import apply_transformations, check_transform

import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    """
    Quick command lines
    --model_path=model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5
    --model_path=model_zoo/segmentation/hair/model_001/checkpoint.hdf5
    --model_path=model_zoo/segmentation/hair/model_002/gaelkt_HairNets.hdf5
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', help='The path to the Keras model file (usually *.hdf5) ')
    parser.add_argument('--export_path', default=None, help='The path for the script result output')
    parser.add_argument('--export_model_name', default=None, help='The model file name')
    parser.add_argument('--save_optimized', default=True, help='The model file name')
    parser.add_argument('--test_config', default=None, help='The model file name')
    args = parser.parse_args()
    inference_mode = 'NCHW'

    model_path = args.model_path
    export_path = args.export_path
    export_model_name = args.export_model_name
    save_optimized = args.save_optimized
    if export_path is None:
        export_path = os.path.dirname(model_path)
    if export_model_name is None:
        export_model_name = '.'.join(os.path.basename(model_path).split('.')[:-1])

    model_ext = model_path.split('.')[-1]
    if model_ext.lower() == 'json':
        from tensorflow.keras.models import model_from_json
        keras_model_in = model_from_json(open(model_path).read())
        keras_model_in.load_weights(model_path.replace(model_ext, 'hdf5'))
    else:
        keras_model_in = load_model(model_path, custom_objects={'tf': tf, 'Functional': Model})
    keras_model = apply_transformations(keras_model_in)
    if save_optimized:
        keras_model.save('.'.join(model_path.split('.')[:-1]) + '_optimized.hdf5')

    adapted_keras_model = adapt_keras_model(keras_model, export_model_name)
    target_shape = keras_model.input_shape[1:3]
    x_in_item = get_test_item_mean_std(target_shape)

    print('\n' + ('='*20) + 'Overall Transform' + ('='*20))
    check_transform(keras_model_in, adapted_keras_model)
    string_list, weight_list = conver_model(adapted_keras_model)
    save_config(string_list, weight_list, adapted_keras_model.name, export_path)