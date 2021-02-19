import ncnn
import numpy as np

out_config_path = 'faceDetector_from_va_320_240_simple_opt.param'
out_weights_path = 'faceDetector_from_va_320_240_simple_opt.bin'

net = ncnn.Net()
net.load_param(out_config_path)
net.load_model(out_weights_path)

num_threads = 4
error_th = 1.e-5
ex = net.create_extractor()
ex.set_num_threads(num_threads)

y_size, x_size, c_size = 320, 320, 3
in_layer_list = [layer.name for layer in net.layers if layer.type == 'Input']
for item in in_layer_list:
    frame = np.random.uniform(0, 255, size=(y_size, x_size, c_size)).astype(np.uint8)
    mat_in = ncnn.Mat.from_pixels(frame, ncnn.Mat.PixelType.PIXEL_BGR, x_size, y_size)
    ex.input(item, mat_in)

s = 0
mat_out = ncnn.Mat()
for blob in net.blobs:
    ex.extract(blob.name, mat_out)
    np_out = np.array(mat_out)
    s += np.prod(np_out.shape)
    print(blob.name, np_out.shape, np.prod(np_out.shape))

print('>>', s, s*4/(2**20))

# frame = np.random.uniform(0, 255, size=fix_none_in_shape(target_shape)[1:]).astype(np.uint8)
# mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR, src_x, src_y, target_x, target_y)
#
# ...
#
# for blob in net.blobs:
#     print(blob.name)