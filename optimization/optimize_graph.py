from tqdm import tqdm
import numpy as np
from unit_test.helper import fix_none_in_shape
from tensorflow.keras.models import Model

from extra_layers.CustomObjects import extra_custom_objects
from optimization.graph.SeparableConv2D_split import check_SeparableConv2D_transfrom, apply_transform_SeparableConv2D, \
    info_SeparableConv2D
from optimization.graph.Conv2DBatchNormalization_merge import check_Conv2DBatchNormalization, \
    apply_transform_Conv2DBatchNormalization, info_Conv2DBatchNormalization
from optimization.graph.BatchNormalization_DepthwiseConv2D_transform import check_BatchNormalization_DepthwiseConv2D, \
    apply_transform_BatchNormalization_DepthwiseConv2D, info_BatchNormalization_DepthwiseConv2D
from optimization.graph.Conv2DSoftmax_split import check_Conv2DSoftmax_transfrom, \
    apply_transform_Conv2DSoftmax, info_Conv2DSoftmax

from optimization.graph.Conv2DReLU_merge import check_Conv2DReLU, apply_transform_Conv2DReLU, info_Conv2DReLU
from optimization.graph.Conv2DSigmoid_merge import check_Conv2DSigmoid, apply_transform_Conv2DSigmoid, info_Conv2DSigmoid
from optimization.graph.Conv2DActivation_merge import check_Conv2DActivation, apply_transform_Conv2DActivation, info_Conv2DActivation
from optimization.graph.ReLU_max_split import check_ReLU_max_transfrom, apply_transform_ReLU_max, info_ReLU_max
from optimization.graph.ActivationReLU_max_split import check_ActivationReLU_max_transfrom, apply_transform_ActivationReLU_max, info_ActivationReLU_max

from optimization.graph.DropLayer import check_DropLayer, \
    apply_transform_DropLayer, info_DropLayer


info_list = [info_DropLayer,
             info_ReLU_max,
             info_ActivationReLU_max,
             info_SeparableConv2D,
             info_Conv2DBatchNormalization,
             info_BatchNormalization_DepthwiseConv2D,
             info_Conv2DReLU,
             info_Conv2DSigmoid,
             info_Conv2DActivation,
             info_Conv2DSoftmax,
             ]
check_transform_list = [check_DropLayer,
                        check_ReLU_max_transfrom,
                        check_ActivationReLU_max_transfrom,
                        check_SeparableConv2D_transfrom,
                        check_Conv2DBatchNormalization,
                        check_BatchNormalization_DepthwiseConv2D,
                        check_Conv2DReLU,
                        check_Conv2DSigmoid,
                        check_Conv2DActivation,
                        check_Conv2DSoftmax_transfrom,
                        ]
apply_transform_list = [apply_transform_DropLayer,
                        apply_transform_ReLU_max,
                        apply_transform_ActivationReLU_max,
                        apply_transform_SeparableConv2D,
                        apply_transform_Conv2DBatchNormalization,
                        apply_transform_BatchNormalization_DepthwiseConv2D,
                        apply_transform_Conv2DReLU,
                        apply_transform_Conv2DSigmoid,
                        apply_transform_Conv2DActivation,
                        apply_transform_Conv2DSoftmax,
                        ]


def transfer_weights(src_model, dst_model, weight_transfer_rule_dict):
    print('\nWeight transfer...')
    for dst_layer in tqdm(dst_model.layers):
        if dst_layer.name in weight_transfer_rule_dict:
            transfer_rule = weight_transfer_rule_dict[dst_layer.name]
            func = transfer_rule['transfer_call']
            func(src_model, dst_model, transfer_rule)
        else:
            dst_layer.set_weights(src_model.get_layer(dst_layer.name).get_weights())

def check_transform(src_model, dst_model, debug=True):
    if debug:
        print("Checking Transfer :: Random value check")
    if type(src_model.input_shape) == list:
        x_in = [np.random.uniform(size=fix_none_in_shape(item)) for item in src_model.input_shape]
    else:
        _shape = tuple(32 if item is None else item for item in src_model.input_shape[1:])
        x_in = np.random.uniform(size=(1,) + _shape)
    dst_output = dst_model.predict(x_in)
    src_output = src_model.predict(x_in)
    if isinstance(dst_output,list):
        for _i, (src_item, dst_item) in enumerate(zip(src_output, dst_output)):
            transform_error = np.abs(src_item - dst_item).mean()
            if debug:
                print(f"  Output {_i}    Transform Error (is less 1e-5) :: {transform_error} , {transform_error < 1e-5}")
    else:
        transform_error = np.abs(dst_output - src_output).mean()
        if debug:
            print(f"         Transform Error (is less 1e-5) :: {transform_error} , {transform_error < 1e-5}")
    return transform_error


def apply_transformations(in_model):
    src_model = None
    src_model_config = in_model.get_config()
    k = 1024
    for info_txt, check_func, apply_func in zip(info_list[:k], check_transform_list[:k], apply_transform_list[:k]):
        if check_func(src_model_config):
            if src_model is None:
                print('Preparation for the transformation...\n')
                src_model = Model.from_config(in_model.get_config(), custom_objects=extra_custom_objects)
                transfer_weights(in_model, src_model, {})
                check_transform(in_model, src_model)

            print(info_txt)
            src_model_config = src_model.get_config()
            dst_model_config, weight_transfer_rule_dict = apply_func(src_model_config)
            dst_model = Model.from_config(dst_model_config, custom_objects=extra_custom_objects)
            transfer_weights(src_model, dst_model, weight_transfer_rule_dict)
            check_transform(in_model, dst_model)

            del src_model
            src_model = dst_model
        else:
            print(f'Nothing to do with {info_txt.split(":")[0]} transform\n')

    if src_model is None:
        return in_model

    return src_model
