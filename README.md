# keras2ncnn
## Description
Export Keras model to Tencent/NCNN.
## Requirements installation
The code was tested with python3.7 with TensorFlow 1.x. The code should work with python3.x . The behaviour with TensorFlow 2.x is unclear.
```
git clone https://github.com/azeme1/keras2ncnn.git
cd keras2ncnn
pip3 install -r requirements.txt 
```
## Usage
The model zoo folder contains the sample model 
([CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5](./model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5)) 
as well as the result of the conversion 
(graph: [CelebA_PrismaNet_256_hair_seg_model_opt_001.param](./model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.param) and 
weights: [CelebA_PrismaNet_256_hair_seg_model_opt_001.bin](./model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.bin))
```
python3 keras2ncnn.py --model_path=model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5
```
## Useful Links
### Tencent/NCNN documentation
* https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step
* https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
* https://github.com/Tencent/ncnn/wiki/operation-param-weight-table
### Graph visualization 
* https://github.com/lutzroeder/netron

## TODO List
### Code
* Tensorflow 2.x support
* Layer Fusion
* Auto compile python bindings for the project
* Add Padding support
### Upcoming Models 
* [https://github.com/thangtran480/hair-segmentation]
* [https://github.com/ItchyHiker/Hair_Segmentation_Keras]
* [https://github.com/gaelkt/HairNets]
* [https://github.com/JungUnYun/Hair_segmentation]
* [https://github.com/YawenSun9/deep-hair-segmentation]
* [https://github.com/mostafa-shalaby84/hair_segmentation_matting]