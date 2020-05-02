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

if __name__ == "__main__":
    l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
    N = len(l)
    arr = np.asarray(l, np.float32)
    arrf = np.fft.fft(arr)
    print(arrf[:N/2 + 1])
