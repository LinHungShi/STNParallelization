import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import pyopencl as cl
import pyopencl.array
import numpy as np
import cv2
import math
import sys
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
 
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# IMG_WIDTH is the width of the transformed image
# IMG_LENGTH is the length of the transformed image
# FETCH_WIDTH is the tile width of the original image which equals the width of transformed image in original image
# FETCH_HEIGHT is the tile HEIGHT of the original image which equals the HEIGHT of transformed image in original image
# x_offset : y intercept of the image
# y_offset : x intercept of the image
'''
ex:
IMG_WIDTH = IMG_HEIGHT = 32
we only fetch the 16x16 image where (0,0) is located at the center of the image.
FETCH_WIDTH = 16
FETCH_HEIGHT = 16
x_offset = 16
y_offset = 16
'''
scaling_bs_kernel = """
#define IMG_WIDTH %(iw)d
#define IMG_HEIGHT %(ih)d
#define FETCH_WIDTH %(fw)d
#define FETCH_HEIGHT %(fh)d

__kernel void scalingBilinearSampling(
			__global float *img,
			__global float *img_loc,
 			__global float *trans_img
 			) {

	unsigned int idx = get_global_id(0);
	float U[FETCH_HEIGHT][2];
	float x_loc = img_loc[2*idx];
	float y_loc;
	float xu_diff;
	float xl_diff;
	float yl_diff;
	float yu_diff;
	int k;
	int x_lbound = (int)x_loc;
	int x_ubound = (int)x_loc + 1;
	int y_lbound;
	xl_diff = 1 - (x_loc - x_lbound);
	xu_diff = 1 - xl_diff;  

	for (k = 0; k < FETCH_HEIGHT; k++) {
		U[k][0] = img[FETCH_WIDTH * k + x_lbound];
		U[k][1] = img[FETCH_WIDTH * k + x_ubound];
	}
		
	for (k = 0; k < IMG_HEIGHT; k++) {	
		y_loc = img_loc[2*(idx+k*IMG_WIDTH)+1];
		y_lbound = (int)y_loc;
		yl_diff =  1 - (y_loc - y_lbound);
		yu_diff = 1 - yl_diff;
		trans_img[k*IMG_WIDTH+idx] = (U[y_lbound][0] * xl_diff + U[y_lbound][1] * xu_diff) * yl_diff 								   
								   + (U[y_lbound+1][0] * xl_diff + U[y_lbound+1][1] * xu_diff) * yu_diff;
		
	}
}
"""
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
start = time.time()
x_start = int(math.floor(trans_loc[0,0]))
x_end = int(math.ceil(trans_loc[-1,0]))
y_start = int(math.floor(trans_loc[0,1]))
y_end = int(math.ceil(trans_loc[-1,1]))

fetch_img = np.array(img[y_start:(y_end+1), x_start:(x_end+1)], dtype = np.float32)
height, width = fetch_img.shape
img_gpu = cl.array.to_device(queue, fetch_img)
trans_loc[:,0] = trans_loc[:,0] - x_start
trans_loc[:,1] = trans_loc[:,1] - y_start
img_loc_gpu = cl.array.to_device(queue, trans_loc)
trans_img_gpu = cl.array.zeros(queue, (row*col), np.float32)
scaling_BS = cl.Program(ctx, scaling_bs_kernel % {"iw":col, "ih":row, "fw":width, "fh": height}).build().scalingBilinearSampling
def scaling_bs(series, exp_time):
	opencl_time = np.zeros((len(series), exp_time))
	for i, t in enumerate(series):
		for j in range(0,exp_time):
			img_gpu = cl.array.to_device(queue, fetch_img)
			start = time.time()
			scaling_BS(queue, (col,1), (1,1), img_gpu.data, img_loc_gpu.data, trans_img_gpu.data)
			opencl_time[i,j] = time.time() - start
	trans_img = trans_img_gpu.get()
	plt.figure()
	plt.imshow(trans_img.reshape(row,col), cmap = 'gray')
	plt.savefig("opencl_img_scaling_{0}.jpg".format(sys.argv[6]))
	return opencl_time
if __name__ == '__main__':
        print "Execute Opencl Bilinear Sampling"
        series = np.array(np.arange(1, int(sys.argv[1])+1)) * int(sys.argv[2])
        exp_times = int(sys.argv[3])
        times = scaling_bs(series, exp_times)
        print "average time: ", np.mean(times,axis = 1)
