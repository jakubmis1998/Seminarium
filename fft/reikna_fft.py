#!/usr/bin/env python
# -*- coding: utf-8 -*-

import reikna.cluda as cluda
from reikna.fft import FFT
import numpy as np
import time

"""
FFT przy użyciu REIKNA(OpenCL)
N - potęga dwójki, w przeciwnym razie dopełnić zerami
Nic nie dała próba bez pętli
"""

# Wybranie api
api = cluda.ocl_api()
thr = api.Thread.create()

sizes = [2**4, 2**8, 2**12, 2**16, 2**20, 2**24, 2**28]
l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
N = len(l)

print("REIKNA FFT")

for size in sizes:
    data = (size / 16) * l
    N = size

    # Przygotowanie danych
    data = np.asarray(data, np.complex64)  # wymagana tablica zespolona
    result_cpu = np.empty(N//2 + 1, np.complex64)

    # Początek pomiaru
    start = time.time()

    # Przygotowanie danych na GPU
    gpu_data = thr.to_device(data)
    gpu_out = thr.to_device(result_cpu)

    # FFT
    fft = FFT(gpu_data)  # przygotowanie
    call_FFT = fft.compile(thr)  # kompilacja
    call_FFT(input=gpu_data, output=gpu_out)  # wywołanie

    # Zwrócenie wyniku
    result_cpu = gpu_out.get()

    # Koniec pomiaru
    print("N = %d\t\t\tTime = %.7f s" % (N, time.time() - start))

    # print(result_gpu)

"""
REIKNA FFT
N = 16                  Time = 0.0168841 s
N = 256                 Time = 0.3132219 s
N = 4096                Time = 0.5725729 s
Traceback (most recent call last):
  File "reikna_fft.py", line 40, in <module>
    call_FFT = dot.compile(thr)  # kompilacja
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/reikna/core/computation.py", line 207, in compile
    self._tr_tree, translator, thread, fast_math, compiler_options, keep).finalize()
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/reikna/core/computation.py", line 559, in finalize
    dependencies=dependent_buffers)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/reikna/cluda/api.py", line 405, in temp_array
    dependencies=dependencies)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/reikna/cluda/tempalloc.py", line 77, in array
    shape, dtype, strides=strides, offset=offset, nbytes=nbytes, allocator=allocator)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/reikna/cluda/api.py", line 173, in array
    return self.thread_cls.array(self, *args, **kwds)
AttributeError: type object 'instance' has no attribute 'array'
"""