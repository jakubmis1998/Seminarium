# Use OpenCL To Add Two Random Arrays (This Way Hides Details)

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy tools
import time

# Initialize the Context
platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)  # Instantiate a Queue

# Initialize array size
N = 10000

# Create two random numpy arrays
a_cpu = np.random.rand(N).astype(np.float32)
b_cpu = np.random.rand(N).astype(np.float32)
# Create numpy destination array 
s = time.time()
result_cpu = np.add(a_cpu, b_cpu)
print("CPU: %.7f s" % (time.time() - s))

gpu_time = time.time()

# Create two random pyopencl arrays
a_gpu = pycl_array.to_device(queue, a_cpu)
b_gpu = pycl_array.to_device(queue, b_cpu) 
# Create an empty pyopencl destination array 
result_gpu = pycl_array.empty_like(a_gpu)

# Create the OpenCL program
program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *result)
{
  int i = get_global_id(0);
  result[i] = a[i] + b[i];
}
""").build()

# Enqueue the program for execution and store the result in result_gpu
event = program.sum(queue, a_gpu.shape, None, a_gpu.data, b_gpu.data, result_gpu.data)
event.wait()

print("GPU: %.7f s" % (time.time() - gpu_time))

print("A: {}".format(a_cpu))
print("B: {}".format(b_cpu))
print("CPU: {}".format(result_cpu))
print("GPU: {}".format(result_gpu))  
