import nvgpu

info = nvgpu.gpu_info()[0]
print(info['type'])
print(info['mem_used_percent'])