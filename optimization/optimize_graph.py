from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Model
from optimization.graph.SeparableConv2D_split import check_SeparableConv2D_transfrom, apply_transform_SeparableConv2D, \
    info_SeparableConv2D
from optimization.graph.Conv2DBatchNormalization_merge import check_Conv2DBatchNormalization, \
    apply_transform_Conv2DBatchNormalization, info_Conv2DBatchNormalization
from optimization.graph.Conv2DReLU_merge import check_Conv2DReLU, apply_transform_Conv2DReLU, info_Conv2DReLU
from optimization.graph.BatchNormalization_DepthwiseConv2D_transform import check_BatchNormalization_DepthwiseConv2D, \
    apply_transform_BatchNormalization_DepthwiseConv2D, info_BatchNormalization_DepthwiseConv2D

info_list = [info_SeparableConv2D, info_Conv2DBatchNormalization,
             info_BatchNormalization_DepthwiseConv2D, info_Conv2DReLU]
check_transform_list = [check_SeparableConv2D_transfrom, check_Conv2DBatchNormalization,
                        check_BatchNormalization_DepthwiseConv2D, check_Conv2DReLU]
apply_transform_list = [apply_transform_SeparableConv2D, apply_transform_Conv2DBatchNormalization,
                        apply_transform_BatchNormalization_DepthwiseConv2D, apply_transform_Conv2DReLU]


def transfer_weights(src_model, dst_model, weight_transfer_rule_dict):
    print('\nWeight transfer...')
    for dst_layer in tqdm(dst_model.layers):
        if dst_layer.name in weight_transfer_rule_dict:
            transfer_rule = weight_transfer_rule_dict[dst_layer.name]
            func = transfer_rule['transfer_call']
            func(src_model, dst_model, transfer_rule)
        else:
            dst_layer.set_weights(src_model.get_layer(dst_layer.name).get_weights())

def check_transform(src_model, dst_model):
    print("Checking Transfer :: Random value check")
    x_in = np.random.uniform(size=(1,) + src_model.input_shape[1:])
    transform_error = np.abs(dst_model.predict(x_in) - src_model.predict(x_in)).mean()
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
                src_model = Model.from_config(in_model.get_config())
                transfer_weights(in_model, src_model, {})
                check_transform(in_model, src_model)

            print(info_txt)
            src_model_config = src_model.get_config()
            dst_model_config, weight_transfer_rule_dict = apply_func(src_model_config)
            dst_model = Model.from_config(dst_model_config)
            transfer_weights(src_model, dst_model, weight_transfer_rule_dict)
            check_transform(in_model, src_model)

            del src_model
            src_model = dst_model
        else:
            print(f'Nothing to do with {info_txt.split(":")[0]} transform\n')

    if src_model is None:
        return in_model

    return src_model
