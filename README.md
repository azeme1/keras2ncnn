# keras2ncnn
## Description
Export Keras model to Tencent/NCNN.
### Supported Keras layer List
* InputLayer
* ReLU/ReLU6/LeakyReLU/Softmax/Sigmoid
* Clip
* MaxPooling2D/AveragePooling2D/MaxPool2D/AvgPool2D
* BatchNormalization
* Conv2D/DepthwiseConv2D/SeparableConv2D
* Concatenate/Add/Multiply
* UpSampling2D(nearest neighbour/ bilinear)
* BatchNormalization(In Progress :: Fusion with Convolution)
* BatchNormalization(In Progress :: Fusion with Convolution)
* ZeroPadding2D(In Progress :: Fusion with Convolution/Pooling)

### Unit tests is written  
* Unit tests with  [Python NCNN inference - pyncnn](https://github.com/caishanli/pyncnn) installed 

### Preconverted models
Some 'preconverted' models can be downloaded from 
[DropBox](https://www.dropbox.com/sh/8anok3k3jxjj81i/AADWMLad_V0MKs4ySN2mgPPda?dl=0)

## Requirements installation
The code was tested with python3.7 with TensorFlow 1.x/2.x (CPU). The code should work with python3.x . 
The behaviour with TensorFlow GPU/TPU.
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
Load the model from the '<file_name>.hdf5' file
```
python3 keras2ncnn.py --model_path=model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5
```
Load the model from the '<file_name>.json' file (the weights should be located at the same folder in '<file_name>.hdf5')
```
python3 keras2ncnn.py --model_path=model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.json
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
* Auto compile python bindings for the project
* Fix layer name length issue
* Export models from Keras applications applications
### Upcoming Models 
* [https://github.com/thangtran480/hair-segmentation]
* [https://github.com/ItchyHiker/Hair_Segmentation_Keras]
* [https://github.com/gaelkt/HairNets]
* [https://github.com/JungUnYun/Hair_segmentation]
* [https://github.com/YawenSun9/deep-hair-segmentation]
* [https://github.com/mostafa-shalaby84/hair_segmentation_matting]

### Thanx
* [https://github.com/cvzakharchenko]