This file lists all the files included in the directory and explains the commands that run the programs

Affine Transformation:
(1). Files:
There are six files used for blinear sampling:
tiled_pyopencl.py : Implements the tiled matrix multiplication by pyopencl.
tiled_pycuda.py : Implements the tiled matrix multiplication by pycuda.
summa_pyopencl.py : Implements SUMMA algorithm by pyopencl
summa_pycuda.py : Implements SUMMA algorithm by pycuda
resultofpyopencl.png : Comparison of time taken by different optimization methods by pyopencl.
resultofpycuda.png : Comparison of time taken by different optimization methods by pycuda.

(2). Result:
Changing the value of the size of image(M*N), the above code will output different results of matrix multiplication and calculate the running time.Then we can use the running time to make plots.

Bilinear Sampling:
(1). Files:
There are seven files used for blinear sampling:
opencl_bs.py : Contains helper function to call basic bilinear sampling opencl kernel function
opencl_scaling_bs1.py : Contains helper function to call optimized 1 bilinear sampling opencl kernel function
opencl_scaling_bs2.py : Contains helper function to call optimized 2 bilinear sampling opencl kernel function
cuda_scaling_bs2.py : Contains helper function to call optimized 2 bilinear sampling cuda kernel function
main_func.py : Main function to call all helper functions
img2.jpg : an image file
lion.jpg : an image file

(2). Command to call function:
python main_func.py <number of threads per block> <multiplier> <repeat time> <scaling ratio> <translation coefficient> <input image name>

<number of threads per block> an integer > 0, create a range for experiment with different number of threads. The total number of threads is a create via np.arange(<number of thread_per_block>) multplied by <multiplier>

<multiplier> an integer > 0, a multiplier used to create the number of threads per block

<repeat time> an integer > 0, the number of experiment times for each case. It is used to calculate the mean computation time every kernel function

<scaling ratio> a ratio > 0 that used for affine transformation

<translation coefficient> a scalar smaller than the number of columns and rows of the image. Used to calculate the translation in affine transformation

<input image name> indicate which input image should be used. Give the name on the input without the extension  

ex: python main_func.py 10 2 10 0.5 200 lion

(3).Result:
Each execution of the command should generate five .jpg files where four of them are the results of transformation and the remaining is the time comparison of different methods

p.s If you set scaling ratio too small, you might suffer the problem of aliasing, which is a common phenomenom after sampling.
