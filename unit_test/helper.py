from tensorflow.keras.models import Model
import os
import numpy as np
import imageio
import cv2
import tensorflow as tf

def clean_name(in_str):
    out_str = "".join(in_str.split()).replace('(', '').replace(')', '').replace(',', 'x').strip()
    return out_str

def tf_random_seed():
    if tf.__version__.split('.')[0] == '1':
        tf.random.set_random_seed(7)
    else:
        tf.random.set_seed(7)

def get_test_item_zero_one(target_shape):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'', 'unit_test_data/person_001_1024x1204.jpg')
    frame_data = imageio.imread(file_path)
    frame_data = cv2.resize(frame_data, tuple(reversed(target_shape[:2])))/255.
    return frame_data[None,...]

def get_test_item_mean_std(target_shape):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'', 'unit_test_data/person_001_1024x1204.jpg')
    frame_data = imageio.imread(file_path)
    frame_data = cv2.resize(frame_data, tuple(reversed(target_shape[:2])))/255.
    frame_data = (frame_data - frame_data.mean()) / frame_data.std()
    return frame_data[None,...]

def save_layer_unit_test(model, model_name, root_folder, inference_mode, get_test_item, dtype=np.float32):
    target_shape = model.input_shape[1:3]
    x_in = get_test_item(target_shape)
    unit_test_path = os.path.join(root_folder, '', model_name, '', 'unit_test')
    os.makedirs(unit_test_path, exist_ok=True)
    dtype_string = str(dtype.__name__)
    for layer_index, layer in enumerate(model.layers):
        if layer_index == 0:
            # TODO:: FIX input name in NCNN
            y_out = x_in.copy().astype(dtype)
            layer_unit_test_path = os.path.join(unit_test_path, '', f'{model.input_names[0]}.{dtype_string}')
        elif layer.name in model.output_names:
            pass
            # TODO:: FIX output name in NCNN
            # layer_unit_test_path = os.path.join(unit_test_path, '', f"{'output'}.{dtype_string}")
            # test_model = Model(model.input, layer.output)
            # y_out = test_model.predict(x_in)
        else:
            layer_unit_test_path = os.path.join(unit_test_path, '', f'{layer.name}.{dtype_string}')
            test_model = Model(model.input, layer.output)
            y_out = test_model.predict(x_in)
        if isinstance(y_out, list):
            y_out = y_out[0]
        if inference_mode == 'NCHW':
            y_out = np.transpose(y_out, (0, 3, 1, 2))
        y_out.astype(dtype).tofile(layer_unit_test_path)

def save_config(string_list, weight_list, model_name, root_folder, dtype=np.float32, debug=True):
    out_config_path = os.path.join(root_folder, '', f'{model_name}.param')
    out_weights_path = os.path.join(root_folder, '', f'{model_name}.bin')

    if debug:
        for item in string_list:
            print(item)

    with open(out_config_path, 'w') as f:
        for item in string_list:
            f.write(item + '\n')

    with open(out_weights_path, 'wb') as f:
        for item in weight_list:
            layer_class, weight_list_item = item
            if layer_class in ['Conv2D', 'DepthwiseConv2D', 'Conv2DTranspose', 'Dense']:
                f.write(np.array([0], dtype=np.uint32).tobytes())
            for weight_array in weight_list_item:
                f.write(weight_array.astype(dtype).tobytes())

    return out_config_path, out_weights_path