# keras2ncnn
## Export Keras model to Tencent/NCNN.
### Supported Keras layers and Features
* Functional Model API (the Sequential and Model as Layer features should be transformed in to flat Functional API Model)
* InputLayer
* ReLU/ReLU6/LeakyReLU/Softmax/Sigmoid
* Clip
* Reshape/Flatten(converted with NCNN::Reshape)
* MaxPooling2D/AveragePooling2D/MaxPool2D/AvgPool2D
* BatchNormalization
* Conv2D/DepthwiseConv2D/SeparableConv2D(converted with split into NCNN::ConvolutionDepthWise->NCNN::Convolution)
* Concatenate/Add/Multiply
* UpSampling2D(nearest neighbour/bilinear)
* BatchNormalization(In Progress :: Fusion with Convolution)
* BatchNormalization(In Progress :: Fusion with Convolution)
* ZeroPadding2D(In Progress :: Fusion with Convolution/Pooling)
* Conv2DTranspose(only for the even strides)

### Unit tests is written  
* Unit tests with  [Python NCNN inference - pyncnn](https://github.com/caishanli/pyncnn) installed 
* Latest models tested tf_nightly-2.5.0.dev20201130-cp37-cp37m-win_amd64.whl / tf_nightly_gpu-2.5.0.dev20201130-cp37-cp37m-win_amd64.whl
* Latest NCNN revision https://github.com/Tencent/ncnn/commit/25b224479cbe535ce35ca92556c8f17d9b9f1951
* Latest PyNCNN revision https://github.com/caishanli/pyncnn/commit/63c77b8bd75dae8e2601b918e92a5050a3fee8df

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
* [https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step]
* [https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure]
* [https://github.com/Tencent/ncnn/wiki/operation-param-weight-table]
* [https://github.com/MarsTechHAN/keras2ncnn]
### Graph visualization 
* [https://github.com/lutzroeder/netron]

## TODO List
### Code
* Add support for the Dense Layer 
* Fix layer name length issue
* Export models from Keras applications applications
* Model in Model support
* Sequential API support
* Mixed mode API support  
### Upcoming Models 
* [https://github.com/thangtran480/hair-segmentation]
* [https://github.com/ItchyHiker/Hair_Segmentation_Keras]
* [https://github.com/gaelkt/HairNets]
* [https://github.com/JungUnYun/Hair_segmentation]
* [https://github.com/YawenSun9/deep-hair-segmentation]
* [https://github.com/mostafa-shalaby84/hair_segmentation_matting]

### Thanx
* [https://github.com/cvzakharchenko]
* [https://github.com/nvoronetskiy]

## Important note
Sometimes good result can be achieved with Tensorflow conversion approach

```
import tensorflow as tf
model = tf.keras.models.load_model("model.h5")
model.save("saved_model")
```

after that convert 'model.ckpt' or 'model.pb' with following scripts
* [https://github.com/Tencent/ncnn/tree/master/tools/mlir]
* [https://github.com/Tencent/ncnn/tree/master/tools/tensorflow]
