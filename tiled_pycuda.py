
from __future__ import division
from scipy import io
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void MatrixMulKernel(float *A, float *B, float *C)
{

  const uint a_width = %(a_width)s;
  const uint b_width = %(b_width)s;  
  
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  const uint Row = by * %(BLOCK_SIZE)s +ty;
  const uint Col = bx * %(BLOCK_SIZE)s +tx;

  float Csub = 0;

  for ( int m=0; m<(a_width-1)/%(BLOCK_SIZE)s+1; ++m)
  {
     
      __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
     
      __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
    if (m*%(BLOCK_SIZE)s+tx<a_width && Row<a_width)
  {  As[ty][tx] = A[Row*a_width +m*%(BLOCK_SIZE)s+tx];}
  
    else 
     {As[ty][tx]=0.0f;}
     if (Col<b_width && (m*%(BLOCK_SIZE)s+ty)<a_width)
      {Bs[ty][tx] = B[Col+(m*%(BLOCK_SIZE)s+ty)*b_width];}
      else 
      {Bs[ty][tx]=0.0f;}
     
      __syncthreads();

    
      for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
        Csub += As[ty][k] * Bs[k][tx];
  
      __syncthreads();
    }

  // Write the block sub-matrix to global memory;
  if (Row<a_width && Col<b_width)
 //{ atomicAdd(&C[Row*b_width+Col],Csub);}
   {C[Row*b_width+Col]=Csub;}
}
"""


##M=1024
##N=1024
time22=range(5)
j=0
for M in [100,320,640,960,1024]:
    N=M
##    a_width=3
##    #b_width
##    target_width=M*N
##    # define size of blocks and tiles sub-matrix 
##    TILE_SIZE = 32
##    BLOCK_SIZE = TILE_SIZE
##
##    a=[]
##    for i in range(M):
##        a=a+[i]*N
##    b=range(N)*M
##    c=[1]*M*N
##    target_cpu=np.array([a,b,c]).astype(np.float32)
##
##    a_cpu = np.random.randn(a_width, a_width).astype(np.float32)
    #2dmatrix_mul
    a_height=2*M
    a_width=3*M
    target_width=N
    TILE_SIZE=3
    BLOCK_SIZE=TILE_SIZE
    a_cpu=np.random.randn(a_height,a_width).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a_cpu)
    target_cpu=np.random.randn(a_width,target_width).astype(np.float32)


    ##target_cpu = np.random.randn(a_width, target_width).astype(np.float32)
    target_gpu = gpuarray.to_gpu(target_cpu)
    c_cpu = np.dot(a_cpu, target_cpu)
    c_gpu = gpuarray.zeros((a_height, target_width), np.float32)

    # get the kernel code from the template 
    # by specifying the constants a_width,b_width and BLOCK_SIZE
    kernel_code = kernel_code_template % { 
        'a_width': a_width,
        'b_width':target_width,
        'BLOCK_SIZE': BLOCK_SIZE,
        }

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")

    matrixmul(
        # inputs
        a_gpu, target_gpu, 
        # output
        c_gpu, 
        # grid of multiple blocks
        grid = (((target_width-1) // TILE_SIZE)+1,((a_height-1)// TILE_SIZE)+1),
        # block of multiple threads
        block = (TILE_SIZE, TILE_SIZE, 1), 
        )
    time2=range(1,11)
    for i in range(1,11):
        start=time.time()
    # call the kernel on the card
        matrixmul(
        # inputs
            a_gpu, target_gpu, 
        # output
            c_gpu, 
        # grid of multiple blocks
            grid = (((target_width-1) // TILE_SIZE)+1,((a_height-1)// TILE_SIZE)+1),
        # block of multiple threads
            block = (TILE_SIZE, TILE_SIZE, 1), 
            )
        time2[i-1]=(time.time()-start)*1000
    time22[j]=np.average(time2)
    print "pycuda time:"
    print time22
    print "python time:"
    start=time.time()
    c_cpu = np.dot(a_cpu, target_cpu)
    print time.time()-start
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
    print "-" * 80
    print "Matrix c (CPU):"
    print c_cpu

    print "-" * 80
    print "Matrix C (GPU):"
    print c_gpu.get()

    print np.allclose(c_cpu, c_gpu.get())
    j=j+1
    ##io.savemat('c_gpu.mat',{'c_gpu':c_gpu.get()})
print time22
