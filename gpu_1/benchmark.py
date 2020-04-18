#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np, time
import skcuda.linalg as linalg
linalg.init()

dim = 100
a = np.random.rand(dim, dim).astype(np.float32) # array

# GPU wkopiowanie danych
start = time.time()
a_gpu = gpuarray.to_gpu(a) # gpuarray
print('Copy in', time.time() - start)

# CPU pomnozenie macierzy
start = time.time()
rescpu = np.dot(a, a) # array
print('CPU:', time.time() - start)

# GPU pomnozenie macierzy
start = time.time()
resgpu = linalg.dot(a_gpu, a_gpu) # gpuarray
print('GPU:', time.time() - start)

# GPU wykopiowanie danych
start = time.time()
resgpu = resgpu.get() # array
print('Copy out', time.time() - start)
print(rescpu - resgpu)
print(np.allclose(rescpu, resgpu))


"""
FLOAT32
bledy rzędu e^-4 / e^-6
szybkie wykopiowanie

FLOAT64
błędy rzędu e^-13 / e^-11
bardzo wolne wykopiowanie

N = 100
CPU : 0.00122 s.
Copy in : 0.00066 s.
GPU : 0.32092 s.
Copy out : 0.00012 s.
True

N = 10 000
CPU : 2.091 s.
Copy in : 0.063 s.
GPU : 0.427 s.
Copy out : 0.140 s.
True

N = 20 000
CPU : 15.121 s.
Copy in : 0.261 s.
GPU : 2.453 s.
Copy out : 0.583 s.
False

N = 30 000
CPU : 70.330 s.
Copy in : 0.584 s.
GPU : 8.379 s.
Copy out : 1.330 s.
False
"""
