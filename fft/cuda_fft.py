#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import pycuda.driver as cuda
import h5py

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
    with h5py.File("/home/jakubmis1998/Seminarium/hdf5/plik.hdf5", "r") as hdf_file:
        
        print("SKCUDA FFT")
        for i in range(len(hdf_file)):
            l = hdf_file[hdf_file.keys()[(i+5)%7]]
            N = len(l)
            arr = np.array(l)
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