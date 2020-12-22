from gpuinfo import GPUInfo
import numpy as np

# print(GPUInfo.gpu_usage())

a = np.random.randint(0, 256, (3, 5))
b = np.empty((3, 5, 4))
for i in range(3):
    for j in range(5):
        b[i][j] = np.array([a[i][j], 0, 0, 255])
print(b)
print(b.shape)
