import numpy as np
import cuda_scaling_bs2
import opencl_scaling_bs2
import opencl_scaling_bs1
import opencl_bs
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
if __name__ == '__main__':
        print "Execute Bilinear Sampling"
        series = np.array(np.arange(1, int(sys.argv[1])+1)) * int(sys.argv[2])
        exp_times = int(sys.argv[3])
        times_opencl2 = opencl_scaling_bs2.scaling_bs2(series, exp_times)
	times_opencl1 = opencl_scaling_bs1.scaling_bs(series, exp_times)
	times_cuda = cuda_scaling_bs2.cuda_bs2(series, exp_times)
	times_opencl = opencl_bs.opencl_bs(series,exp_times)
	print "times_opencl2", times_opencl2	
	print "times_opencl1", times_opencl1
	print "times_opencl", times_opencl
	print "times_cuda", times_cuda
	plt.figure()
	plt.plot(series, np.mean(times_opencl2, axis = 1).flatten(), 'r--',label = 'opecl_optimized2')
	plt.plot(series, np.mean(times_opencl1, axis = 1).flatten(), 'k--',label = 'opecl_optimized1')
	plt.plot(series, np.mean(times_opencl, axis = 1).flatten(), 'b--',label = 'opecl_basic')
	plt.plot(series, np.mean(times_cuda, axis = 1).flatten(), 'g--', label = 'cuda_optimized2')
	plt.legend(loc = 'upper left')
	plt.xlabel("Threads_per_Block")
	plt.ylabel("Time")
	plt.title("Naive v.s Opt1 v.s Opt2")
	plt.savefig("Time_Comparison_{0}_{1}_{2}_{3}_{4}_{5}.jpg".format(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5], sys.argv[6]))
