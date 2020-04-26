# Use OpenCL To multiplicate Two Random Arrays

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
N = 5000

# Create two random numpy arrays
a_cpu = np.random.rand(N, N).astype(np.float32)
b_cpu = np.random.rand(N, N).astype(np.float32)
# Create numpy destination array 
s = time.time()
result_cpu = np.dot(a_cpu, b_cpu)
print("CPU: %.7f s" % (time.time() - s))

gpu_time = time.time()

# Create two random pyopencl arrays
a_gpu = pycl_array.to_device(queue, a_cpu)
b_gpu = pycl_array.to_device(queue, b_cpu) 
# Create an empty pyopencl destination array 
result_gpu = pycl_array.empty_like(a_gpu)

program = cl.Program(context, """
__kernel void matmul(const unsigned int size, __global float * matrix1, __global float * matrix2, __global float * res) {

int i = get_global_id(1); 
int j = get_global_id(0);

res[i + size * j] = 0;

for (int k = 0; k < size; k++)
{
    res[j + size * i] += matrix1[k + size * i] * matrix2[j + size * k];
}

}
""").build()

# Enqueue the program for execution and store the result in result_gpu
event = program.matmul(queue, a_gpu.shape, None, np.int32(N), a_gpu.data, b_gpu.data, result_gpu.data)
event.wait()
result_gpu = result_gpu.get()

print("GPU: %.7f s" % (time.time() - gpu_time))

print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))
