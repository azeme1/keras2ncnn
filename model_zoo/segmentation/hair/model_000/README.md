# Model source
The model taken from [https://github.com/ItchyHiker/Hair_Segmentation_Keras.git].
The model was adapted to current script keras2ncnn.py. If you have [pyncnn](https://github.com/caishanli/pyncnn) installed 
in you environment you should be able to reproduce the inference result 
(see [demo.py](./model_zoo/segmentation/hair/model_000/demo.py)) - with MAE :: 6.6825e-08:

![alt Keras vs NCNN inference result](demo.png?raw=true "Keras vs NCNN inference result")
