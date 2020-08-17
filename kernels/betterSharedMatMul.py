#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

"""
Zoptymalizowane mnozenie macierzy z pamiecia dzielona
Dowolne N - sprawdzanie w kernelu przy kafelkowaniu
MNOZENIE MACIERZY NA GPU Z PAMIECIA DZIELONA
"""

mod = SourceModule("""
    __global__ void multiplicate_matrixes(float* A, float* B, float* C, int N)
{
    float CValue = 0;

    const int TILE_DIM = 16;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + N - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < N && Row < N)
             As[threadIdx.y][threadIdx.x] = A[Row*N + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < N && Col < N)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*N + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int k = 0; k < TILE_DIM; k++)
             CValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

         __syncthreads();
    }

    if (Row < N && Col < N)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}
""")
multiplicate_matrixes = mod.get_function("multiplicate_matrixes")
start = cuda.Event()
end = cuda.Event()

N = 15
print("N = {}".format(N))
a = np.random.randn(N, N).astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)
result = gpuarray.empty((N, N), np.float32)

start.record()
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

# GRID 2D - by wszystkie pary były dostępne np. 2 bloki - (0,0) (0,1) ... (0, 31) ... (31, 31)
multiplicate_matrixes(a_gpu, b_gpu, result, np.int32(N), block=(16, 16, 1), grid=((N+15)//16, (N+15)//16, 1))

result_gpu = result.get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPU: \t%.7f s" % secs)

s = time.time()
result_cpu = np.dot(a, b)
print("CPU: \t%.7f s" % (time.time() - s))

print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))

"""
N = 16
GPU:    0.0016329 s
CPU:    0.0000682 s

N = 64
GPU:    0.0016500 s
CPU:    0.0001140 s

N = 128
GPU:    0.0015208 s
CPU:    0.0012839 s

N = 256
GPU:    0.0012313 s
CPU:    0.0020812 s

N = 512
GPU:    0.0020073 s
CPU:    0.0038199 s

N = 1024
GPU:    0.0065504 s
CPU:    0.0095088 s

N = 2048
GPU:    0.0313431 s
CPU:    0.0397780 s

N = 4096
GPU:    0.1855716 s
CPU:    0.2085080 s

N = 8192
GPU:    1.5270277 s
CPU:    1.2801330 s

N = 16384
GPU:    13.2325605 s
CPU:    8.3152699 s

"""
