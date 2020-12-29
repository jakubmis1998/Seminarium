from rest_framework import viewsets, status
from rest_framework.response import Response
from django.http import FileResponse, HttpResponse
import io, os, json, random, string
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# PyCuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Computing
import numpy as np
from math import sqrt
import tifffile
import psutil
import nvidia_smi

# Benchmark
import time

# GPU initializations
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
import pycuda.autoinit
device = cuda.Device(0)

class SystemUsage(viewsets.ViewSet):
    def list(self, request):
        print("READ SYSTEM USAGE")
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        response = HttpResponse(json.dumps({
            'cpu_count': psutil.cpu_count(),
            'ram_usage': dict(psutil.virtual_memory()._asdict()),
            'cpu_usage': psutil.cpu_percent(percpu = True),
            'gpu_usage': np.around(100 * (mem_res.used / mem_res.total), decimals = 1)
        }))
        response["Content-Type"] = 'application/json'
        response.status_code = status.HTTP_200_OK
        print("RETURN SYSTEM USAGE")
        return response

def id_generator(size, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class KernelProcessing(viewsets.ViewSet):
    def create(self, request):
        ctx = device.make_context()

        # Kernel
        # m może zawierać kilka stron.
        # Każdy wątek oblicza jeden element result z jednej strony m - czyli wątków jest X * Y.
        # Każdy wątek oblicza sobie dwie ostatnie pętle.
        mod = SourceModule("""
        #define MAX(a, b) (a)<(b)?(b):(a)
        #define MIN(a, b) (a)>(b)?(b):(a)
        // Funkcja zwracajaca pierwiastek z liczby typu float
        __device__ float my_sqrt(float x)
        {
            return sqrt(x);
        }
        __device__ int getHalfRing(int R, int dx)
        {
            int abs_dx = abs(dx);
            return int(ceil(my_sqrt(R*R - abs_dx*abs_dx)));
        }
        __global__ void gpu_int_mask_multi_thread(const int X, const int Y, const int R, const int T, const int *mask, const int *m, int *result)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza macierzy m i result
            int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny macierzy m i result
            // X * Y * pageNumber
            int pageId = X * Y * blockIdx.z;

            int x, y, dx, ry, drx, dry, from, to, lxbound, rxbound, p, c;
            if ((row < X) && (col < Y)) {
                lxbound = MAX(0, row - R);
                rxbound = MIN(X, row + R + 1);
                p = 0;
                c = 0;
                for(x = lxbound; x < rxbound; x++) {
                    dx = row - x;
                    drx = dx + R;
                    // 1 SPOSOB (R - liczone po kwadracie)
                    //ry = R;

                    // 2 SPOSOB (getHalfRing, ry jako int)
                    ry = getHalfRing(R, dx);

                    from = MAX(0, col - ry + 1);
                    to = MIN(Y, col + ry);
                    for(y = from; y < to; y++) {
                        dry = y + R - col;

                        // If
                        if ((x - row) * (x - row) + (y - col) * (y - col) < R * R) {
                            int maskValue = mask[drx * (2*R+1) + dry];
                            p += maskValue;
                            if (m[pageId + (x * Y + y)] + T <= m[pageId + (row * Y + col)]) {
                                c += maskValue;
                            }
                        }

                        // Przesuniecie bitowe
                        //p += mask[drx * (2*R+1) + dry];
                        //c += ((((m[row * Y + col] - T) - m[x * Y + y]) >> 31) * mask[drx * (2*R+1) + dry]);

                        // Przepisanie obrazka
                        // result[row * Y + col] = m[row * Y + col];
                    }
                }
                result[pageId + (row * Y + col)] = (256 * c) / p;
            }
        }
        """)
        gpu_int_mask_multi_thread = mod.get_function("gpu_int_mask_multi_thread")

        # Get parameters from request data
        parameters = json.loads(request.data.get("processing_info"))
        filename = id_generator(10) + parameters.get("filename")
        method = parameters.get("method")
        pages = parameters.get("pages")
        X = int(parameters.get("X"))
        Y = int(parameters.get("Y"))
        R = int(parameters.get("R"))
        T = int(parameters.get("T"))

        # Save image and load to tifffile, then remove from disk
        path = default_storage.save(filename, ContentFile(request.data["image"].read()))
        img = tifffile.imread(filename)
        os.remove(filename)

        # Mask and result array
        mask = [[1 if (R - i)*(R - i) + (R - j)*(R - j) <= R * R else 0 for i in range(2*R+1)] for j in range(2*R + 1)]
        mask = np.array(mask, dtype=np.int32)

        # Zmienne do pomiaru czasu na GPU
        start = cuda.Event()
        end = cuda.Event()
        time = 0

        # # Dane GPU
        start.record()
        # All pages
        m_gpu = gpuarray.to_gpu(img.astype(np.int32))
        mask_gpu = gpuarray.to_gpu(mask)
        result_gpu = gpuarray.empty((pages, Y, X), dtype=np.int32)

        # # Wywołanie kernel'a
        gpu_int_mask_multi_thread(
            np.int32(X),
            np.int32(Y),
            np.int32(R),
            np.int32(T),
            mask_gpu,
            m_gpu,
            result_gpu,
            block=(32, 32, 1),
            grid=((X+31)//32, (X+31)//32, pages)
        )

        # Wynik GPU
        result_gpu_kernel = result_gpu.get()
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        time += secs

        tmp_name = f"{id_generator(15)}.tif"
        # Zapisanie przerobionego pliku na dysk
        for i in range(pages):
            tifffile.imwrite(tmp_name, result_gpu_kernel[i].astype(img.dtype), append=True, compression='deflate')

        # Free gpu memory
        ctx.pop()
        del mask_gpu
        del m_gpu
        del result_gpu

        # Odczytanie go w formie binarnej
        with open(tmp_name, 'rb') as fp:
            data = fp.read()
        os.remove(tmp_name)
        print("time: ", time)

        response = FileResponse(
            io.BytesIO(data),
            as_attachment=True
        )
        response["Content-Type"] = 'multipart/form-data'
        response.status_code = status.HTTP_200_OK
        return response
