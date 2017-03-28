import time
import matplotlib
import opencl_scaling_bs1
import opencl_bs
import cuda_scaling_bs2
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
 			__global float *trans_img,
 			__global int* bins
 			) {
		unsigned int ldx = get_local_id(0);
		unsigned int tile_size = get_local_size(0);
		unsigned int gid = get_group_id(0);
		__local float U[FETCH_HEIGHT][2];
		float y_loc;
		float xu_diff;
		float xl_diff;
		float yl_diff;
		float yu_diff;
		int k;
		int x_lbound = gid;
		int x_ubound = x_lbound + 1;
		int y_lbound;
		int gdx;
		int tmp = ldx;
		while (tmp < FETCH_HEIGHT) {
			U[tmp][0] = img[FETCH_WIDTH * tmp + x_lbound];
			U[tmp][1] = img[FETCH_WIDTH * tmp + x_ubound];
			tmp += tile_size;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		tmp = ldx;
		float x_loc;
		int basics = 0;
		for (unsigned int i = 0; i < gid;i++) {
			basics = basics + bins[i];
			
		}
		
		while (tmp < bins[gid]) {
			
			gdx = tmp + basics;
			x_loc = img_loc[2*gdx];
			xl_diff = 1 - (x_loc - x_lbound);
			xu_diff = 1 - xl_diff;  	
			for (k = 0; k < IMG_HEIGHT; k++) {	
				y_loc = img_loc[2*(gdx+k*IMG_WIDTH)+1];
				y_lbound = (int)y_loc;
				yl_diff =  1 - (y_loc - y_lbound);
				yu_diff = 1 - yl_diff;
				trans_img[k*IMG_WIDTH+gdx] = (U[y_lbound][0] * xl_diff + U[y_lbound][1] * xu_diff) * yl_diff + (U[y_lbound+1][0] * xl_diff + U[y_lbound+1][1] * xu_diff) * yu_diff;		
			
			}
			tmp += tile_size;
		}
}
"""
def countNumElem(x, bins):
	elem_array = np.zeros(bins).astype(np.int32)
	for i in x:
		if i < bins:
			elem_array[int(math.floor(i))] += 1
		elif i == bins:
			elem_array[bins-1] += 1
	return elem_array

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
x_start = int(math.ceil(trans_loc[0,0]))
x_end = int(math.ceil(trans_loc[-1,0]))
y_start = int(math.ceil(trans_loc[0,1]))
y_end = int(math.ceil(trans_loc[-1,1]))
fetch_img = np.array(img[y_start:(y_end+1), x_start:(x_end+1)], dtype = np.float32)
img_gpu = cl.array.to_device(queue, fetch_img)
trans_loc[:,0] = trans_loc[:,0] - x_start
trans_loc[:,1] = trans_loc[:,1] - y_start
img_loc_gpu = cl.array.to_device(queue, trans_loc)
fetch_height, fetch_width = fetch_img.shape
trans_img_gpu = cl.array.zeros(queue, (row*col), np.float32)
bins = countNumElem(trans_loc[0:col,0], fetch_width)
bins_gpu = cl.array.to_device(queue, bins)
scaling_BS = cl.Program(ctx, scaling_bs_kernel% {"iw":np.uint32(col), "ih":np.uint32(row), "fw":np.uint32(fetch_width), "fh": np.uint32(fetch_height)}).build().scalingBilinearSampling
#repeat_time = int(sys.argv[2])
#multiplier = int(sys.argv[3])
#series = np.array(np.arange(1, repeat_time+1))*multiplier
#exp_time = int(sys.argv[4])
#opencl_times = np.zeros((len(series),exp_time)).astype(np.float32)
#opencl_times1 = opencl_scaling_bs1.scaling_bs(series, exp_time)
#opencl_times2 = opencl_bs.opencl_bs(series, exp_time)
#cuda_times = cuda_scaling_bs2.cuda_bs2(series, exp_time)
def scaling_bs2(series, exp_time):
	opencl_times = np.zeros((len(series), exp_time))
	for i, t in enumerate(series):
		for j in range(exp_time):
			trans_img_gpu = cl.array.zeros(queue, (row*col), np.float32)
			start = time.time()
			scaling_BS(queue, (fetch_height*t,1), (t,1), img_gpu.data, img_loc_gpu.data, trans_img_gpu.data, bins_gpu.data)
			opencl_times[i, j] = time.time() - start
	trans_img = trans_img_gpu.get()
	plt.figure()
	plt.imshow(trans_img.reshape(row,col), cmap = 'gray')
	plt.savefig("opencl_img_scaling2_{0}.jpg".format(sys.argv[6]))
	return opencl_times
#print "opencl average time: ", np.mean(opencl_times, axis = 1)
#print "opencl1 average time: ", np.mean(opencl_times1,axis = 1)
#print "opencl2 average time: ", np.mean(opencl_times2,axis = 1)
#print "cuda average time: ", np.mean(cuda_times, axis = 1)
#print np.array(len(series)).shape
#print np.mean(opencl_times, axis = 1).shape
#plt.plot(series, np.mean(opencl_times, axis = 1).flatten(), 'r--',label = 'opecl_optimized2')
#plt.plot(series, np.mean(opencl_times1, axis = 1).flatten(), 'k--',label = 'opecl_optimized1')
#plt.plot(series, np.mean(opencl_times2, axis = 1).flatten(), 'b--',label = 'opecl_basic')
#plt.plot(series, np.mean(cuda_times, axis = 1).flatten(), 'g--', label = 'cuda_optimized2')
#plt.legend(loc = 'upper right')
#plt.savefig("opencl_compare.jpg")
#print "img_loc", img_loc
#trans_img = trans_img_gpu.get()
#print "scaling trans img:", trans_img.reshape(row, col)
#plt.figure()
#plt.imshow(trans_img.reshape(row,col), cmap = 'gray')
#plt.savefig("img_scaling2.jpg")

if __name__ == '__main__':
        print "Execute Opencl Bilinear Sampling"
        series = np.array(np.arange(1, int(sys.argv[1])+1)) * int(sys.argv[2])
        exp_times = int(sys.argv[3])
        times = scaling_bs2(series, exp_times)
        print "average time: ", np.mean(times,axis = 1)
