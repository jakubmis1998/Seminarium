#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

mod = SourceModule("""
    __global__ void add_arrays(int n, float *a, float *b, float* out)
    {
        //Id bloku wymiar bloku(256) numer watku
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n){
            out[i] = a[i] + b[i];
        }
    }
""")
add_arrays = mod.get_function("add_arrays")
start = cuda.Event()
end = cuda.Event()

N = 100000
print("N = {}".format(N))
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
result = gpuarray.empty(N, np.float32)

start.record()
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Copy in: \t%.7f s" % secs)

start.record()
add_arrays(np.int32(N), a_gpu, b_gpu, result, block=(256, 1, 1), grid=((N+255)//256, 1, 1))
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPU: \t\t%.7f s" % secs)

start.record()
result_gpu = result.get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Copy out: \t%.7f s" % secs)

s = time.time()
result_cpu = a + b
print("CPU: \t\t%.7f s" % (time.time() - s))

print("Computation error: {}".format(abs(result_cpu - result_gpu)))
