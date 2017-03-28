from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np
import time
import scipy
from scipy import signal
import matplotlib
import cv2
import sys
import math
matplotlib.use('Agg')
from matplotlib import pyplot as plt

scaling_bs_kernel = """
#include "stdio.h"
#define IMG_WIDTH %(iw)d
#define IMG_HEIGHT %(ih)d
#define FETCH_WIDTH %(fw)d
#define FETCH_HEIGHT %(fh)d
__global__ void scalingBilinearSampling(
			const float* img,
			const float* img_loc,
 			float* trans_img,
 			const int* __restrict__ bins
 			) {
		int ldx = threadIdx.x;
		int tile_size = blockDim.x;
		int gid = blockIdx.x;
		__shared__ float U[FETCH_HEIGHT][2];
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
		__syncthreads();
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
				trans_img[k*IMG_WIDTH+gdx] = (U[y_lbound][0] * xl_diff + U[y_lbound][1] * xu_diff)* yl_diff + (U[y_lbound+1][0] * xl_diff + U[y_lbound+1][1] * xu_diff) * yu_diff;		
			}
			tmp += tile_size;
		}
}
"""
#printf(\"ldx=%d, k = %d, tmp = %d, basics = %d, tsize = %d, gdx = %d, index: %d\\n\", ldx, k, tmp, basics, tile_size, gdx, k*IMG_WIDTH+gdx);
#trans_img[k*IMG_WIDTH+gdx] = 0;
def countNumElem(x, bins):
	elem_array = np.zeros(bins).astype(np.int32)
	#print "x: ", x
	for i in x:
		if i < bins:
			#print "{0} put in {1}".format(i, int(math.floor(i)))
			elem_array[int(math.floor(i))] += 1
		elif i == bins:
			elem_array[bins-1] += 1
	return elem_array

# Read Image
img = cv2.imread('{0}.jpg'.format(sys.argv[6]))[:,:,0].astype(np.float32)
#img = np.ones((10,10)).astype(np.float32)
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

x_start = int(math.ceil(trans_loc[0,0]))
x_end = int(math.ceil(trans_loc[-1,0]))
y_start = int(math.ceil(trans_loc[0,1]))
y_end = int(math.ceil(trans_loc[-1,1]))

fetch_img = np.array(img[y_start:(y_end+1), x_start:(x_end+1)], dtype = np.float32)
img_gpu = gpuarray.to_gpu(fetch_img)
trans_loc[:,0] = trans_loc[:,0] - x_start
trans_loc[:,1] = trans_loc[:,1] - y_start
fetch_height, fetch_width = fetch_img.shape
#print "trans_loc: ", trans_loc
#print "x_start: ", x_start
#print "y_start: ", y_start
#print "x_end: ", x_end
#print "y_end: ", y_end
print "fetch height, width:", fetch_img.shape
img_loc_gpu = gpuarray.to_gpu(trans_loc)
trans_img_gpu = gpuarray.zeros(shape=(row*col), dtype = np.float32)
bins = countNumElem(trans_loc[0:col,0], fetch_width)
bins_gpu = gpuarray.to_gpu(bins)
#% {"iw":col, "ih":row, "fw":fetch_width
prg = compiler.SourceModule(scaling_bs_kernel % {"iw":col, "ih":row, "fw":fetch_width, "fh":fetch_height})
scalingBS = prg.get_function('scalingBilinearSampling')
threads_per_block = int(sys.argv[2])
print "bins:", bins
def cuda_bs2(series, exp_time):
	trans_img_gpu = gpuarray.zeros(shape=(row*col), dtype = np.float32)
	cuda_time = np.zeros((len(series), exp_time))
	for i,t in enumerate(series):
		for j in range(exp_time):
			trans_img_gpu = gpuarray.zeros(shape=(row*col), dtype = np.float32)
			start = time.time()
			scalingBS(
				img_gpu,
				img_loc_gpu,
				trans_img_gpu,
				bins_gpu,
				block = (t,1,1),
				grid = (fetch_width,1,1)
				)			
			cuda_time[i,j] = time.time() - start
	trans_img = trans_img_gpu.get()
	print "trans_img:", trans_img
	plt.figure()
	plt.imshow(trans_img.reshape(row,col), cmap = 'gray')
	plt.savefig("cuda_img_scaling2_{0}.jpg".format(sys.argv[6]))
	return cuda_time
#print "time: ", time.time() - start
#trans_img = trans_img_gpu.get()
#print "cuda scaling trans img:", trans_img.reshape(row, col)
#plt.figure()
#plt.imshow(trans_img.reshape(row,col), cmap = 'gray')
#plt.savefig("cuda_img_scaling2.jpg")

if __name__ == '__main__':
	print "Execute Cuda Bilinear Sampling"
	series = np.array(np.arange(1, int(sys.argv[1])+1)) * int(sys.argv[2])
	print series
	exp_times = int(sys.argv[3])
	times = cuda_bs2(series, exp_times)
	print "average time: ", np.mean(times,axis = 1)
