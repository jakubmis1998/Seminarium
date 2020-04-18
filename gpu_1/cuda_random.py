import pycuda.autoinit
import pycuda.driver as drv
import pycuda.curandom as curandom
import numpy


size = 100

#start timer
start = drv.Event()
end = drv.Event()
start.record()

#cuda operation which fills the array with random numbers
curandom.rand((size, ))

#stop timer
end.record()
end.synchronize()

#calculate used time
secs = start.time_till(end)*1e-3

print("GPU: %.7fs" % secs)

#start timer
start = drv.Event()
end = drv.Event()
start.record()

#cpu operation which fills the array with random data
numpy.random.rand(size).astype(numpy.float32)

#stop timer
end.record()
end.synchronize()

#calculate used time
secs = start.time_till(end)*1e-3
print("CPU: %.7fs" % secs)
"""
N = 10
CPU: 0.0000479s
GPU: 0.2225941s

N = 100
CPU: 0.0000602s
GPU: 0.2135181s

N = 1 000
CPU: 0.0000700s
GPU: 0.2158139s

N = 10 000
CPU: 0.0003818s
GPU: 0.2096904s

N = 1 000 000
CPU: 0.0245770s
GPU: 0.2100595s

N = 100 000 000
CPU: 1.3161676s
GPU: 0.2197666s

N = 1 000 000 000
CPU: 12.7776494s
GPU: 0.2498580s
"""