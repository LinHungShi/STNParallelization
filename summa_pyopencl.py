#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
Image Convolution in PyOpenCL
"""
import numpy as np
#import PYOPENCL modules and libraries
import pyopencl as cl
import pyopencl.array
import sys
#the following module is used to mark the time stamps
import time
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
#  Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
 
# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)



kernel = """
__kernel void MatrixMulKernel(__global float *A, __global float *B, __global float *C)
{
    int y=get_global_id(0);
    int x=get_global_id(1);
//  __local float Csub[2][1];
    float temp=0.0f;
    int ty=get_local_id(0);
    int tx=get_local_id(1);

    for (int k=0;k<%(L)d;k+=3)
    {
      float Asub[2][3];
      for (int i=0;i<2;i++)
      {
        for (int j=0;j<3;j++)
        {
            Asub[i][j]=A[(get_group_id(0)*2+i)*%(L)d+k+j];
        }
      }
      float Bsub[3][1];
      for (int i=0;i<3;i++)
      {
        for (int j=0;j<1;j++)
        {
          Bsub[i][j]=B[(k+i)*%(N)d+get_group_id(1)*1+j];
        }
      }
        __syncthreads();      
       barrier(CLK_LOCAL_MEM_FENCE);
           for (int m = 0; m < 3; m++) {
                   temp+= Asub[ty][m]*Bsub[m][tx];
                   
           }
        //   Csub[ty][tx]=temp;
        C[y*%(N)d+x]=temp;
      //     __syncthreads();

   }
   C[y*%(N)d+x]=temp;
 
               //  C[y*%(N)d+x]=Csub[ty][tx];
   //  __syncthreads();
}
"""
##time11=range(5)
##jj=0
##for numblock_row in [100,320,640,960,1024]:
numblock_row=1024
M=2*numblock_row
L=3*numblock_row
N=960


a_cpu = np.random.randn(M, L).astype(np.float32)
a_gpu = cl.array.to_device(queue, a_cpu)



target_cpu = np.random.randn(L, N).astype(np.float32)
target_gpu = cl.array.to_device(queue, target_cpu)
c_cpu = np.dot(a_cpu, target_cpu)
c_gpu = cl.array.zeros(queue, (M, N), np.float32)

prg=cl.Program(ctx,kernel%{'L': L,'N':N}).build()
MatrixMulKernel=prg.MatrixMulKernel
MatrixMulKernel(queue,
                (numblock_row*2,N),
                (2, 1),a_gpu.data, target_gpu.data,c_gpu.data)
time1=range(1,11)
for i in range(1,11):
    start=time.time()
    MatrixMulKernel(queue,
                    (numblock_row*2,N),
                    (2, 1),a_gpu.data, target_gpu.data,c_gpu.data)
    time1[i-1]=time.time()-start
time11=np.average(time1)


##    print "pyopencl time:"
####    print time11*1000
####    print "python time:"
##    start=time.time()
##    c_cpu = np.dot(a_cpu, target_cpu)
####    print time.time()-start
##    ## print the results
##
##    print "-" * 80
##    print "Matrix c (CPU):"
##    print c_cpu
##
##    print "-" * 80
##    print "Matrix C (GPU):"
##    print c_gpu.get()
##
##    print np.allclose(c_cpu, c_gpu.get())
##    jj=jj+1
print time11

