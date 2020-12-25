from gpuinfo import GPUInfo
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%') # percentage usage
# print(GPUInfo.gpu_usage())
