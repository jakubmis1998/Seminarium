#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

a_np = np.random.rand(3).astype(np.float32)
b_np = np.random.rand(3).astype(np.float32)

print("A: {}".format(a_np))
print("B: {}".format(b_np))

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
ctx = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(ctx)  # Create a command queue with your context

mf = cl.mem_flags
a_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(__global const float *a, __global const float *b, __global float *result)
{
  int gid = get_global_id(0);
  result[gid] = a[gid] + b[gid];
}
""").build()

gpu_sum = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_gpu, b_gpu, gpu_sum)

res_gpu = np.empty_like(a_np)
cl.enqueue_copy(queue, res_gpu, gpu_sum)

# Check on CPU with Numpy:
print("CPU: {}".format(a_np + b_np))
print("GPU: {}".format(res_gpu))
