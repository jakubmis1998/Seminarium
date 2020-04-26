# Use OpenCL To Multiplicate Two Random Matrixes

import numpy as np  # Import Numpy tools
import pyviennacl as vienna
import pyviennacl.linalg as linalg  # Import PyViennaCl (Python ViennaCl library for Linear Algebra)
import time
import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array
platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)  # Instantiate a Queue
# Initialize array size
N = 2

# Create two random numpy arrays
a_cpu = np.random.rand(N, N).astype(np.float32)
b_cpu = np.random.rand(N, N).astype(np.float32)
# Create numpy destination array 
s = time.time()
result_cpu = np.dot(a_cpu, b_cpu)
print("CPU: %.7f s" % (time.time() - s))

gpu_time = time.time()

# Create two random pyviennacl matrixes
a_gpu = vienna.Matrix(a_cpu)
b_gpu = vienna.Matrix(b_cpu)
print("copy in: %.7f s" % (time.time() - gpu_time))
# Create an empty pyopencl destination array 
result_gpu = linalg.prod(a_gpu, b_gpu).as_ndarray()
# result_gpu = (a_gpu * b_gpu).as_ndarray()
print("GPU: %.7f s" % (time.time() - gpu_time))

print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))
