import reikna.cluda as cluda
from reikna.fft import FFT
import numpy as np

api = cluda.ocl_api()
thr = api.Thread.create()

l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
N = len(l)

data = np.asarray(l, np.complex64)  # wymagana tablica zespolona

gpu_data = thr.to_device(data)
gpu_out = thr.array(N/2 + 1, dtype=np.complex64)

dot = FFT(gpu_data)
call_FFT = dot.compile(thr)
call_FFT(input=gpu_data, output=gpu_out)

result_gpu = gpu_out.get()

print(result_gpu)