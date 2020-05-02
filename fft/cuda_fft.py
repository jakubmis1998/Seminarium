import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from time import sleep

def pikoFFT(arr):
    x = arr.astype('float32')
    N = len(arr)

    xgpu = gpuarray.to_gpu(x)

    y = gpuarray.empty(N/2 + 1, np.complex64)

    plan_forward = cu_fft.Plan(N, np.float32, np.complex64)

    cu_fft.fft(xgpu, y, plan_forward)

    return y.get()


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
    l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]  # rozmiar tablicy jest potega dwojki
    arr = np.array(l)
    print(pikoFFT(arr))

