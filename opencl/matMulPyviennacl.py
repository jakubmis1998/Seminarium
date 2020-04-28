# Use PyViennaCL To Multiplicate Two Random Matrixes

import numpy as np
import pyviennacl as vienna
import pyviennacl.linalg as linalg  # Import PyViennaCl (Python ViennaCl library for Linear Algebra)
import time

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

a = [[1,2],[3,4]]

# Create two random pyviennacl matrixes
a_gpu = vienna.Matrix(a)
b_gpu = vienna.Matrix(a)
print("copy in: %.7f s" % (time.time() - gpu_time))
# Create an empty pyopencl destination array 
result_gpu = linalg.prod(a_gpu, b_gpu).as_ndarray()
# result_gpu = (a_gpu * b_gpu).as_ndarray()
print("GPU compute: %.7f s" % (time.time() - gpu_time))

print("Computation error:\n {}".format(abs(np.subtract(result_cpu, result_gpu))))
