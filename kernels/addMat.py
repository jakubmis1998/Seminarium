#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

"""
blockDim.x * blockDim.y = rozmiar bloku = 256 wątków (256 always)
blockIdx = numer bloku (0 --- (N*N+255)/256)
blockDim.x * threadIdx.y + threadIdx.x = Który wątek w danym bloku (0 --- 255)
"""

mod = SourceModule("""
    __global__ void add_matrixes(int n, float *a, float *b, float* out)
    {
        // Wielokrotnosc 256 + dany watek w bloku(0-255)
        int i = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y  + threadIdx.x);

        if (i < n*n){
            out[i] = a[i] + b[i];
        }
    }
""")
add_matrixes = mod.get_function("add_matrixes")
start = cuda.Event()
end = cuda.Event()

N = 5000
print("N = {}".format(N))
a = np.random.randn(N, N).astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)
result = gpuarray.empty((N, N), np.float32)

start.record()
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Copy in: \t%.7f s" % secs)

start.record()
add_matrixes(np.int32(N), a_gpu, b_gpu, result, block=(16, 16, 1), grid=((N*N+255)//256, 1, 1))
end.record()
end.synchronize()
#calculate used time
secs = start.time_till(end)*1e-3
print("GPU: \t\t%.7f s" % secs)
# print("GPU: \t\t%.7f s" % (time.time() - s))

start.record()
result_gpu = result.get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Copy out: \t%.7f s" % secs)

s = time.time()
result_cpu = a + b
print("CPU: \t\t%.7f s" % (time.time() - s))

print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))
