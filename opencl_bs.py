import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import pyopencl as cl
import pyopencl.array
import numpy as np
import cv2
import sys
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
 
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)


basic_BS = cl.Program(ctx, """
__kernel void BasicBilinearSampling(__global float *original_img, __global float *imgloc, __global float *trans_img, const unsigned int img_w) {
	unsigned int idx = get_global_id(0);
	int m[2];
	int n[2];
	float x_loc = imgloc[idx*2];
	int x = x_loc;
	float y_loc = imgloc[idx*2+1];
	int y = y_loc;
	m[0] = x;
	m[1] = x_loc - x != 0 ? (x + 1):x;
	n[0] = y;
	n[1] = y_loc - y != 0 ? (y + 1):y;
	float c = 0;
	float x_flag = 1.0;
	float y_flag;
	int x_end = m[0] == m[1] ? 1:2;
	int y_end = n[0] == n[1] ? 1:2;
	for (int i = 0; i < x_end; i++) {
		y_flag = 1.0;
		for (int j = 0; j < y_end; j++) {
			float tmp1 = original_img[n[j]*img_w + m[i]];
			//printf("img = %f, m = %d, n = %d, x_loc = %f, y_loc = %f   \t", tmp1, m[i], n[j], x_loc, y_loc);
			c += original_img[n[j] * img_w + m[i]] * (1 - x_flag * (x_loc - m[i])) * (1 - y_flag * (y_loc - n[j]));
			y_flag = -1.0;
			//printf("idx: %d, c:%f\t", idx, c);
		}
		x_flag = -1.0;
	}
	trans_img[idx] = c;
	//if (idx == 320)
	//	printf("tras_img[%d] = %f", idx, trans_img[idx]);
}
""").build().BasicBilinearSampling

# Read Image
img = cv2.imread('{0}.jpg'.format(sys.argv[6]))[:,:,0].astype(np.float32)
coef = float(sys.argv[4])
trans = float(sys.argv[5])
np.random.seed(123)
affine = np.array([[coef,0,trans],[0, coef,trans]]).astype(np.float32)
row, col = img.shape
tmp_x = np.expand_dims(np.array(range(col)*row).astype(np.float32), axis = 0)
tmp_y = np.expand_dims(np.repeat(range(row), col).astype(np.float32), axis = 0)
constant = np.expand_dims(np.ones(row*col).astype(np.float32),axis = 0)
img_loc = np.concatenate([tmp_x, tmp_y, constant], axis = 0)
trans_loc = np.dot(img_loc.T, affine.T)
img_gpu = cl.array.to_device(queue, img)
img_loc_gpu = cl.array.to_device(queue, trans_loc)
def opencl_bs(series, exp_time):
	opencl_time = np.zeros((len(series), exp_time))
	for i, t in enumerate(series):
		for j in range(exp_time):
			trans_img_gpu = cl.array.zeros(queue, (row*col), np.float32)
			start = time.time()
			basic_BS(queue, (row*col,1), (1,1), img_gpu.data, img_loc_gpu.data, trans_img_gpu.data, np.uint32(col))
			opencl_time[i,j] = time.time() - start
	trans_img = trans_img_gpu.get()
	plt.imshow(trans_img.reshape(row, col), cmap = 'gray')
	plt.savefig("opencl_img_{0}.jpg".format(sys.argv[6]))
	return opencl_time
if __name__ == '__main__':
        print "Execute Opencl Bilinear Sampling"
        series = np.array(np.arange(1, int(sys.argv[1])+1)) * int(sys.argv[2])
        exp_times = int(sys.argv[3])
        times = opencl_bs(series, exp_times)
        print "average time: ", np.mean(times,axis = 1)
