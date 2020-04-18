import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas as cublas

N = 2

# A = np.array(([1, 2, 3], [4, 5, 6]), order = 'F').astype(np.float32)
a = np.random.randn(N, N).astype(np.float32)
A = np.asarray(a, order = 'F')
# B = np.array(([7, 8, 1, 5], [9, 10, 0, 9], [11, 12, 5, 5]), order = 'F').astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)
B = np.asarray(a, order = 'F')

A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)

C_gpu = gpuarray.empty((N, N), np.float32)

alpha = np.float32(1.0)
beta  = np.float32(0.0)

cublas_handle = cublas.cublasCreate()
cublas.cublasSgemm(cublas_handle, 'n', 'n', N, N, N, alpha, A_gpu.gpudata, N, B_gpu.gpudata, N, beta, C_gpu.gpudata, N)
cublas.cublasDestroy(cublas_handle)

result_cpu = np.dot(A, B)
result_gpu = C_gpu.reshape(C_gpu.shape, order = 'F').get()

print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))