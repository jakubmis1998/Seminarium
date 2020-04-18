#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

"""
N musi byc wielkrotnoscia BlockDim.x (N = k * BLOCKSIZE.X)
w przypadku innych N mozna dopelnic macierz zerami do N x N
Ilosc kafelkow = ilosc blokow - jeden blok oblicza jeden kafelek
Ilosc watkow = elementow kafelka - jeden watek oblicza jeden element kafelka
MNOZENIE MACIERZY NA GPU Z PAMIECIA DZIELONA
"""


def cpu_matmul(a, b, N):
    """Przykladowe mnozenie macierzy CPU"""
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

        float Cij = 0;
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
""")
multiplicate_matrixes = mod.get_function("multiplicate_matrixes")
start = cuda.Event()
end = cuda.Event()

N = 16384
print("N = {}".format(N))
a = np.random.randn(N, N).astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)
result = gpuarray.empty((N, N), np.float32)

start.record()
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

# GRID 2D - by wszystkie pary były dostępne np. 2 bloki - (0,0) (0,1) ... (0, 31) ... (31, 31)
multiplicate_matrixes(np.int32(N), a_gpu, b_gpu, result, block=(16, 16, 1), grid=((N+15)//16, (N+15)//16, 1))

result_gpu = result.get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPU: \t%.7f s" % secs)

s = time.time()
result_cpu = np.dot(a, b)
# result_cpu = cpu_matmul(a, b, N)
print("CPU: \t%.7f s" % (time.time() - s))

# print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))

"""
CPU - wlasna funkcja mnozenia macierzy
CPU#2 - numpy

N = 16384
GPU:    25.1421484 s
CPU:    8.4353402 s

N = 4096
GPU:    0.4480610 s
CPU:    0.2074392 s

N = 1024
GPU:    0.0105925 s
CPU#2:  0.0144479 s	
CPU:    764.0058229 s

N = 512
GPU:    0.0026974 s
CPU#2:  0.0039580 s
CPU:    96.4422300 s

N = 256
GPU:    0.0012883 s
CPU#2:  0.0024741 s
CPU:    12.2167108 s

N = 128
GPU:    0.0017900 s
CPU#2:  0.0026271 s
CPU:    1.6160641 s

N = 64
GPU:    0.0016289 s
CPU#2:  0.0001109 s
CPU:    0.2495339 s

N = 16
GPU:    0.0015827 s
CPU#2:  0.0000670 s
CPU:    0.0082412 s
"""