#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

"""
NAIWNY ALGORYTM MNOZENIA MACIERZY NA GPU Z PAMIECIA GLOBALNA

CPU
1 2 3
4 5 6
7 8 9
GPU
1 2 3 4 5 6 7 8 9

row - ktory wiersz przetwarza dany watek
col - ktora kolumne przetwarza dany watek

row * N + col ---> wielokrotnosc wiersza + kolumna. 5 = 3 * 1 + 2
row * N + k ---> wielkrotnosc wiersza + 0, 1, 2 - caly wiersz(kolejno trójki)
k * N + col ---> wielokrotnosc kolumny + N, 2N, 3N - cala kolumna(co trzy)
"""

mod = SourceModule("""
    __global__ void multiplicate_matrixes(int N, float *a, float *b, float* out)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza
        int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny

        if ((row < N) && (col < N)){
            // Each thread loads one row of M and one column of N, 
            // to produce one element of out.
            for (int k = 0; k < N; k++) {
                out[row * N + col] += a[row * N + k] * b[k * N + col];
            }
        }
    }
""")
multiplicate_matrixes = mod.get_function("multiplicate_matrixes")
start = cuda.Event()
end = cuda.Event()

N = 128
print("N = {}".format(N))
a = np.random.randn(N, N).astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)
result = gpuarray.empty((N, N), np.float32)

# COPY IN
start.record()
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

# GRID 2D - by wszystkie pary były dostępne np. 2 bloki - (0,0) (0,1) ... (0, 31) ... (31, 31)
multiplicate_matrixes(np.int32(N), a_gpu, b_gpu, result, block=(32, 32, 1), grid=((N+31)//32, (N+31)//32, 1))

# COPY OUT
result_gpu = result.get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPU: \t%.7f s" % secs)

# CPU + NUMPY
s = time.time()
result_cpu = np.dot(a, b)
print("CPU: \t%.7f s" % (time.time() - s))

# print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))

"""
N = 16384
GPU:            32.8151660 s
CPU:            8.4335532 s

N = 4096
GPU:            0.5388198 s
CPU:            0.2132671 s

N = 1024
GPU:            0.0133108 s
CPU:            0.0147789 s

N = 128
GPU:            0.0015333 s
CPU:            0.0024779 s
"""
