import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import time
import pycuda.driver as cuda
import skcuda.cublas as cublas

start = cuda.Event()
end = cuda.Event()
N = 4096
print("N = {}".format(N))

a = np.random.randn(N, N).astype(np.float32)
A = np.asarray(a, order = 'F')

b = np.random.randn(N, N).astype(np.float32)
B = np.asarray(a, order = 'F')

# GPU
start.record()
A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)
C_gpu = gpuarray.empty((N, N), np.float32)

alpha = np.float32(1.0)
beta  = np.float32(0.0)

cublas_handle = cublas.cublasCreate()
cublas.cublasSgemm(cublas_handle, 'n', 'n', N, N, N, alpha, A_gpu.gpudata, N, B_gpu.gpudata, N, beta, C_gpu.gpudata, N)
cublas.cublasDestroy(cublas_handle)

result_gpu = C_gpu.reshape(C_gpu.shape, order = 'F').get()

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPU: \t\t%.7f s" % secs)

# CPU NUMPY
s = time.time()
result_cpu = np.dot(A, B)

print("CPU: \t\t%.7f s" % (time.time() - s))

# print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))

"""
N = 128
GPU:            0.0010433 s
CPU:            0.0014701 s

N = 512
GPU:            0.0023028 s
CPU:            0.0041699 s

N = 1024
GPU:            0.0056924 s
CPU:            0.0142150 s

N = 2048
GPU:            0.0184509 s
CPU:            0.0498800 s

N = 4096
GPU:            0.0823461 s
CPU:            0.2100949 s

N = 10000
GPU:            0.5869617 s
CPU:            2.0249629 s
"""