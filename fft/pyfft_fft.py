#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyfft.cl import Plan
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

"""
FFT przy użyciu PYFFT
N - potęga dwójki, w przeciwnym razie dopełnić zerami
"""

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)  # Create a command queue with your context

sizes = [2**4, 2**8, 2**12, 2**16, 2**20, 2**24, 2**28]
l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
N = len(l)

print("PYFFT FFT")

for size in sizes:
    data = (size / 16) * l
    N = size

    # Przygotowanie danych
    arr = np.array(data)

    # Przygotowanie danych
    data_cpu = np.array(data, np.complex64)  # Wymagana tablica zespolona
    result_cpu = np.empty(N/2 + 1, np.complex64)

    # Początek pomiaru
    start = time.time()

    # Plan
    plan = Plan(N, queue=queue, context=context)

    # Przygotowanie danych na GPU
    data_gpu = cl_array.to_device(queue, data_cpu)
    result_gpu = cl_array.to_device(queue, result_cpu)

    # FFT
    plan.execute(data_in=data_gpu.data, data_out=result_gpu.data)

    # Zwrócenie danych
    result_cpu = result_gpu.get()

    # Koniec pomiaru
    print("N = %d\t\t\tTime = %.7f s" % (N, time.time() - start))

    # print(result)

# Wait for finish = True
"""
PYFFT FFT
N = 16                  Time = 0.0123880 s
N = 256                 Time = 0.0048120 s
N = 4096                        Time = 0.0056698 s
N = 65536                       Time = 0.0089841 s
N = 1048576                     Time = 0.0123160 s
Traceback (most recent call last):
  File "pyfft_fft.py", line 48, in <module>
    plan.execute(data_in=data_gpu.data, data_out=result_gpu.data)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/pyfft/plan.py", line 271, in _executeInterleaved
    batch, data_in, data_out)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/pyfft/plan.py", line 256, in _execute
    self._context.wait()
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/pyfft/cl.py", line 114, in wait
    self._queue.finish()
pyopencl._cl.LogicError: clFinish failed: INVALID_COMMAND_QUEUE
"""

"""
PYFFT FFT
N = 16                  Time = 5.0155160 s
N = 256                 Time = 5.0166249 s
N = 4096                        Time = 5.0173211 s
N = 65536                       Time = 5.0199249 s
N = 1048576                     Time = 5.0243330 s
Traceback (most recent call last):
  File "pyfft_fft.py", line 54, in <module>
    result_cpu = result_gpu.get()
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/pyopencl/array.py", line 721, in get
    ary, event1 = self._get(queue=queue, ary=ary, async_=async_, **kwargs)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/pyopencl/array.py", line 682, in _get
    wait_for=self.events, is_blocking=not async_)
  File "/home/jakubmis1998/.local/lib/python2.7/site-packages/pyopencl/__init__.py", line 1719, in enqueue_copy
    return _cl._enqueue_read_buffer(queue, src, dest, **kwargs)
pyopencl._cl.RuntimeError: clEnqueueReadBuffer failed: OUT_OF_RESOURCES
"""