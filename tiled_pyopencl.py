
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

  const uint a_width = %(a_width)s;
  const uint b_width = %(b_width)s;  
  
  const uint bx = get_group_id(1);
  const uint by = get_group_id(0);

  const uint tx = get_local_id(1);
  const uint ty = get_local_id(0);

  const uint Row = get_global_id(0);
  const uint Col =get_global_id(1);
  //bx * %(BLOCK_SIZE)s +tx;

  float Csub = 0;

  for ( int m=0; m<(a_width-1)/%(BLOCK_SIZE)s+1; ++m)
  {
     
      __local float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
     
      __local float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
    if (m*%(BLOCK_SIZE)s+tx<a_width && Row<a_width)
  {  As[ty][tx] = A[Row*a_width +m*%(BLOCK_SIZE)s+tx];}
  
    else 
     {As[ty][tx]=0.0f;}
     if (Col<b_width && (m*%(BLOCK_SIZE)s+ty)<a_width)
      {Bs[ty][tx] = B[Col+(m*%(BLOCK_SIZE)s+ty)*b_width];}
      else 
      {Bs[ty][tx]=0.0f;}
     
      __syncthreads();
      barrier(CLK_LOCAL_MEM_FENCE);
    
      for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
        Csub += As[ty][k] * Bs[k][tx];
  
      __syncthreads();
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  // Write the block sub-matrix to global memory;
  if (Row<a_width && Col<b_width)

   {C[Row*b_width+Col]=Csub;}
}
"""
time_python=range(5)
time11=range(5)
jj=0
a_width=3
for M in [100,320,640,960,1024]:
    N=M
    ##M=1024
    ##N=1024
    #b_width
    target_width=M*N
    # define size of blocks and tiles sub-matrix 
    TILE_SIZE = 3
    BLOCK_SIZE = TILE_SIZE

    a=[]
    for i in range(M):
        a=a+[i]*N
    b=range(N)*M
    c=[1]*M*N
    target_cpu=np.array([a,b,c]).astype(np.float32)

    a_cpu = np.random.randn(a_width, a_width).astype(np.float32)
    a_gpu = cl.array.to_device(queue, a_cpu)



    ##target_cpu = np.random.randn(a_width, target_width).astype(np.float32)
    target_gpu = cl.array.to_device(queue, target_cpu)
    c_cpu = np.dot(a_cpu, target_cpu)
    c_gpu = cl.array.zeros(queue, (a_width, target_width), np.float32)
    # get the kernel code from the template 
    # by specifying the constants a_width,b_width and BLOCK_SIZE

    prg=cl.Program(ctx,kernel%{'a_width': a_width,'b_width':target_width,'BLOCK_SIZE': BLOCK_SIZE}).build()
    MatrixMulKernel=prg.MatrixMulKernel

    # get the kernel function from the compiled module
    ##MatrixMulKernel(queue,
    ##                ((((target_width) // TILE_SIZE)+1)*TILE_SIZE,(((a_width) // TILE_SIZE)+1)*TILE_SIZE),
    ##                (TILE_SIZE, TILE_SIZE),a_gpu.data, target_gpu.data,c_gpu.data)
    MatrixMulKernel(queue,
                    (((a_width-1)/TILE_SIZE+1)*TILE_SIZE,((target_width-1)/TILE_SIZE+1)*TILE_SIZE),
                    (TILE_SIZE, TILE_SIZE),a_gpu.data, target_gpu.data,c_gpu.data)
    time1=range(1,11)
    for i in range(1,11):
        start=time.time()
    # call the kernel on the card
    ##MatrixMulKernel(queue,
    ##                ((((target_width-1)/TILE_SIZE)+1)*TILE_SIZE,(((a_width-1)/ TILE_SIZE)+1)*TILE_SIZE),
    ##                (TILE_SIZE, TILE_SIZE),a_gpu.data, target_gpu.data,c_gpu.data)
        MatrixMulKernel(queue,
                        (((a_width-1)/TILE_SIZE+1)*TILE_SIZE,((target_width-1)/TILE_SIZE+1)*TILE_SIZE),
                        (TILE_SIZE, TILE_SIZE),a_gpu.data, target_gpu.data,c_gpu.data)
        time1[i-1]=time.time()-start
    time11[jj]=np.average(time1)*1000#ms
   
    print "pyopencl time:"
    print time1

    print time11[jj]
    print "python time:"
    start=time.time()
    c_cpu = np.dot(a_cpu, target_cpu)
    time_python[jj]=time.time()-start
    
    jj=jj+1

print time11
print time_python
    
    # print the results
    ##print "-" * 80
    ##
    ##print "Matrix A (GPU):"
    ##print a_gpu.get()
    ##
    ##print "-" * 80
    ##print "Matrix target (GPU):"
    ##print target_gpu.get()
    ##
    ##print "-" * 80
    ##print "Matrix c (CPU):"
    ##print c_cpu
    ##
    ##print "-" * 80
    ##print "Matrix C (GPU):"
    ##print c_gpu.get()
    ##
    ##print np.allclose(c_cpu, c_gpu.get())

    ##io.savemat('c_gpu.mat',{'c_gpu':c_gpu.get()})
