from __future__ import division
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
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ float Csub[2][1];
    float temp=0.0f;
    int ty=threadIdx.y;
    int tx=threadIdx.x;

    for (int k=0;k<%(L)d;k+=3)
    {
      float Asub[2][3];
      for (int i=0;i<2;i++)
      {
        for (int j=0;j<3;j++)
        {
            Asub[i][j]=A[(blockIdx.y*2+i)*%(L)d+k+j];
        }
      }
      float Bsub[3][1];
      for (int i=0;i<3;i++)
      {
        for (int j=0;j<1;j++)
        {
          Bsub[i][j]=B[(k+i)*%(N)d+blockIdx.x*1+j];
        }
      }
        __syncthreads();      
   
           for (int m = 0; m < 3; m++) {
                   temp+= Asub[ty][m]*Bsub[m][tx];
                   
           }
           Csub[ty][tx]=temp;
    //C[y*%(N)d+x]=temp;
      //     __syncthreads();

   }
 
                 C[y*%(N)d+x]=Csub[ty][tx];
   //  __syncthreads();
}
"""
time22=range(5)
j=0
for numblock in [100,320,640,960,1024]:
    

    M=2*numblock
    L=3*numblock
    N=numblock


    a_cpu = np.random.randn(M, L).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a_cpu)



    target_cpu = np.random.randn(L, N).astype(np.float32)
    target_gpu = gpuarray.to_gpu(target_cpu)
    c_cpu = np.dot(a_cpu, target_cpu)
    c_gpu = gpuarray.zeros((M, N), np.float32)

    kernel_code = kernel_code_template % { 
        'L':L,
        'N': N
        }

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)

    ### get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")
    matrixmul(
        a_gpu,
        target_gpu, 
        c_gpu,  
        grid = (numblock,numblock),
        block = (1,2,1),
        )
    time2=range(1,11)
    for i in range(1,11):
        start=time.time()
        matrixmul(
            a_gpu,
            target_gpu, 
            c_gpu,  
              grid = (numblock,numblock),
            block = (1,2,1),
            )
        time2[i-1]=time.time()-start
    time22[j]=np.average(time2)
    print "pycuda time:"
    print time22
    print "python time:"
    start=time.time()
    c_cpu = np.dot(a_cpu, target_cpu)
    print time.time()-start
    ## print the results
    print "-" * 80
    print "a (GPU)"
    print a_gpu.get()

    print "-" * 80
    print "Matrix target (GPU):"
    print target_gpu.get()

    print "-" * 80
    print "Matrix c (CPU):"
    print c_cpu

    print "-" * 80
    print "Matrix C (GPU):"
    print c_gpu.get()

    print np.allclose(c_cpu, c_gpu.get())
    j=j+1

print time22
