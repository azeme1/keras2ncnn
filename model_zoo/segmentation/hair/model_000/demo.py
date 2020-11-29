import ncnn
import cv2
import numpy as np
from tensorflow.keras.models import load_model

keras_model = load_model('model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.hdf5')

net = ncnn.Net()
net.load_param('model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.param')
net.load_model('model_zoo/segmentation/hair/model_000/CelebA_PrismaNet_256_hair_seg_model_opt_001.bin')
num_threads = 4

frame_bgr = cv2.imread('./unit_test/unit_test_data/person_001_1024x1204.jpg')
target_y, target_x = (256, 256)
frame_bgr_show = cv2.resize(frame_bgr, (target_x, target_y))
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
src_y, src_x = frame.shape[:2]


input_data = cv2.resize(frame, (target_x, target_y))
mean_value = input_data.reshape((-1, 3)).mean(0)
inv_std_value = 1. / (1.e-5 + input_data.reshape((-1, 3)).std(0))

mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_RGB, src_x, src_y, target_x, target_y)
mat_in.substract_mean_normalize(mean_value, inv_std_value)

keras_in = (np.array(input_data)[None] - mean_value)*inv_std_value
keras_out = keras_model.predict(keras_in)[0, ..., 0]

ex = net.create_extractor()
ex.set_num_threads(num_threads)
ex.input("data_0", mat_in)
mat_out = ncnn.Mat()
ex.extract("output_3lidentity_0", mat_out)
ncnn_out = np.array(mat_out)
ncnn_out = np.transpose(ncnn_out, (1, 2, 0))

out_file_name = 'model_zoo/segmentation/hair/model_000/demo.png'
print(f'MAE :: {np.abs(keras_out.flatten() - ncnn_out.flatten()).mean()}')
print(f'Output saved {out_file_name}')

frame_bgr_show = np.pad(frame_bgr_show, ((7, 7), (7, 7), (0, 0)), constant_values=128)

keras_out_show = cv2.cvtColor((255*keras_out).astype(np.uint8), cv2.COLOR_GRAY2BGR)
keras_out_show = cv2.putText(keras_out_show, 'Keras Result', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
keras_out_show = np.pad(keras_out_show, ((7, 7), (7, 7), (0, 0)), constant_values=128)
ncnn_out_show = cv2.cvtColor((255*ncnn_out).astype(np.uint8), cv2.COLOR_GRAY2BGR)
ncnn_out_show = cv2.putText(ncnn_out_show, 'NCNN Result', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
ncnn_out_show = np.pad(ncnn_out_show, ((7, 7), (7, 7), (0, 0)), constant_values=128)

show_frame = np.hstack([frame_bgr_show, keras_out_show, ncnn_out_show])
cv2.imwrite(out_file_name, show_frame)



