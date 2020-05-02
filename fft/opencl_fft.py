from pyfft.cl import Plan
import numpy
import pyopencl as cl
import pyopencl.array as cl_array

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)

l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
N = len(l)

plan = Plan(N, queue=queue)

data = numpy.asarray(l, numpy.float32)
out = numpy.empty(N/2+1, numpy.complex64)
gpu_data = cl_array.to_device(queue, data)
gpu_out = cl_array.to_device(queue, out)
plan.execute(data_in=gpu_data.data, data_out=gpu_out.data)

result = gpu_out.get()
print(result)