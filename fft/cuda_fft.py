#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import pycuda.driver as cuda

"""
FFT przy użyciu SKCUDA
N - potęga dwójki, w przeciwnym razie dopełnić zerami
"""

def pikoFFT(arr):
    x = arr.astype('float32')
    N = len(arr)

    start = cuda.Event()
    end = cuda.Event()

    # Początek pomiaru
    start.record()

    # Przygotowanie pamięci na GPU
    xgpu = gpuarray.to_gpu(x)
    y = gpuarray.empty(N/2 + 1, np.complex64)

    plan_forward = cu_fft.Plan(N, np.float32, np.complex64)

    # FFT
    cu_fft.fft(xgpu, y, plan_forward)

    # Zwrócenie wyniku
    left = y.get()

    # Koniec pomiaru
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("N = %d\t\t\tTime = %.7f s" % (N, secs))

    # print(left)

    return left


# def pikoBatchFFT(data):
#     x = data.astype('float32')
#     batchSize, N = x.shape
#     print ("batchSize", batchSize, " N ", N)

#     xgpu = gpuarray.to_gpu(x)
#     xf_gpu = gpuarray.empty((batchSize, N//2+1), np.complex64)

#     plan_forward = cu_fft.Plan(N, np.float32, np.complex64, batchSize)
#     cu_fft.fft(xgpu, xf_gpu, plan_forward)

#     print (xf_gpu.get())


if __name__ == '__main__':
    sizes = [2**4, 2**8, 2**12, 2**16, 2**20, 2**24, 2**28]
    l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]

    N = len(l)

    print("SKCUDA FFT")

    for size in sizes:
        data = (size / 16) * l
        N = size

        # Przygotowanie danych
        arr = np.array(data)
        pikoFFT(arr)

"""
SKCUDA FFT
N = 16             Time = 0.3122584 s
N = 256            Time = 0.0010869 s
N = 4 096          Time = 0.0009068 s
N = 65 536         Time = 0.0047716 s
N = 1 048 576      Time = 0.0040942 s
N = 16 777 216     Time = 0.0387470 s
N = 268 435 456    Time = 0.5076792 s
"""