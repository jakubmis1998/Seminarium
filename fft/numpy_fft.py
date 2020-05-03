#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np
# from skcuda.fft import fft, Plan

# N = 16
# x = np.asarray(np.random.rand(N), np.float32)
# xf = np.fft.fft(x)
# x_gpu = gpuarray.to_gpu(x)
# xf_gpu = gpuarray.empty(N/2+1, np.complex64)
# plan = Plan(x.shape, np.float32, np.complex64)
# fft(x_gpu, xf_gpu, plan)
# print(xf)
# print(50*"-")
# print(xf_gpu.get())
# np.allclose(xf[0:N/2+1], xf_gpu.get(), atol=1e-6)

import numpy as np
import time

"""
FFT przy użyciu NUMPY
N - potęga dwójki, w przeciwnym razie dopełnić zerami
"""

if __name__ == "__main__":
    sizes = [2**4, 2**8, 2**12, 2**16, 2**20, 2**24, 2**28]
    l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]

    N = len(l)

    print("NUMPY FFT")

    for size in sizes:
        data = (size / 16) * l
        N = size

        # Przygotowanie danych
        arr = np.asarray(data, np.float32)

        # Pomiar
        start = time.time()
        arrf = np.fft.fft(arr)
        print("N = %d\t\t\tTime = %.7f s" % (N, time.time() - start))

    # print(arrf[:N/2 + 1])

"""
NUMPY FFT
N = 16             Time = 0.0000710 s
N = 256            Time = 0.0000610 s
N = 4 096          Time = 0.0005472 s
N = 65 536         Time = 0.0089750 s
N = 1 048 576      Time = 0.1112919 s
N = 16 777 216     Time = 2.2151301 s
N = 268 435 456    Time = 42.1939871 s
"""