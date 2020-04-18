#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

"""
MNOZENIE MACIERZY NA GPU Z PAMIECIA DZIELONA
"""

def cpu_matmul(a, b, N):
    result = np.empty([N, N], dtype=np.float32)
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += a[i][k] * b[k][j]
            result[i][j] = tmp
    return result

mod = SourceModule("""
    __global__ void multiplicate_matrixes(int N, float *a, float *b, float* out)
    {
        const int TILE_N = 16;
        __shared__ float As[TILE_N][TILE_N];
        __shared__ float Bs[TILE_N][TILE_N];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i < N) && (j < N)){
            float Cij = 0.0f;
            for (int m = 0; m < (N / TILE_N); m++){ /* m-ty kafelek */
                As[tx][ty] = a[i*N + (m*TILE_N + ty )];
                Bs[tx][ty] = b[(m*TILE_N + tx)*N + j];

                __syncthreads(); // zapewnia zaladowanie macierzy

                for (int k = 0; k < TILE_N; k++) {
                    Cij += As[tx][k]*Bs[k][ty];
                }

                __syncthreads(); // zapewnia obliczenia w pelni wykonane
            }

            out[i*N + j] = Cij; // pamiec globalna
        }
    }
""")
multiplicate_matrixes = mod.get_function("multiplicate_matrixes")
start = cuda.Event()
end = cuda.Event()

N = 2
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
# GRID 2D - by wszystkie pary były dostępne np. 2 bloki - (0,0) (0,1) ... (0, 31) ... (31, 31)
multiplicate_matrixes(np.int32(N), a_gpu, b_gpu, result, block=(16, 16, 1), grid=((N+15)//16, (N+15)//16, 1))
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
result_cpu = np.dot(a, b)
print("CPU: \t\t%.7f s" % (time.time() - s))

# print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))

print("\n")
print(result_cpu)
print(result_gpu)
