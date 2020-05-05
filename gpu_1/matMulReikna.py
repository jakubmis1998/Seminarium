import numpy
from numpy.linalg import norm
import reikna.cluda as cluda
from reikna.linalg import MatrixMul
import time

api = cluda.ocl_api()
thr = api.Thread.create()

N = 1000

a = numpy.random.randn(N, N).astype(numpy.float32)
b = numpy.random.randn(N, N).astype(numpy.float32)

start = time.time()
s = time.time()
a_dev = thr.to_device(a)
b_dev = thr.to_device(b)
res_dev = thr.array((N, N), dtype=numpy.float32)
print("copy in: %.7f s" % (time.time() - s))

s = time.time()
dot = MatrixMul(a_dev, b_dev, out_arr=res_dev)  # przygotowanie
print("gpu compute: %.7f s" % (time.time() - s))

s = time.time()
dotc = dot.compile(thr)  # kompilacja
dotc(res_dev, a_dev, b_dev)  # wywolanie
result_gpu = res_dev.get()
print("copy out: %.7f s" % (time.time() - s))
print("GPU: %.7f s" % (time.time() - start))

s = time.time()
result_cpu = numpy.dot(a, b)
print("cpu: %.7f s" % (time.time() - s))

print("Computation error:\n {}".format(abs(numpy.subtract(result_cpu, result_gpu))))

"""
N = 10 000
copy in: 0.1102679 s
gpu compute: 0.0023351 s
copy out: 2.6105001 s
GPU: 2.7232220 s
cpu: 2.0742841 s

N = 1 000
copy in: 0.0007150 s
gpu compute: 0.0007720 s
copy out: 0.1911099 s
GPU: 0.1926558 s
cpu: 0.0144141 s
"""