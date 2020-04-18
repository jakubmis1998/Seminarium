import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from skcuda import linalg
import pycuda.autoinit
import numpy
import time

N = 35000

linalg.init()

start = cuda.Event()
end = cuda.Event()

a_cpu = numpy.random.rand(N, N).astype(numpy.float32)

start.record()
start.synchronize()
a_gpu = gpuarray.to_gpu(a_cpu.astype(numpy.float32))
gpu_result = linalg.dot(a_gpu, a_gpu).get()
end.record()
end.synchronize()
print("GPU: {} s.".format(start.time_till(end)*1e-3))

s = time.time()
cpu_result = numpy.dot(a_cpu, a_cpu)
e = time.time()
print("CPU: {} s.".format(e-s))

print(numpy.allclose(gpu_result, cpu_result))


"""
N = 10
GPU: 0.001 s.
CPU: 0.00003 s.
True

N = 100
GPU: 0.0006 s.
CPU: 0.0057 s.
True

N = 10 000
GPU: 0.661 s.
CPU: 4.665 s.
True

N = 20 000
GPU: 3.759 s.
CPU: 15.431 s.
False

N = 30 000
GPU: 10.921 s.
CPU: 77.448 s.
False

N = 35 000
a_gpu * a_gpu
cuMemAlloc: out of memory
"""