from rest_framework import viewsets, status
from rest_framework.response import Response
from django.http import FileResponse, HttpResponse
import io, os, json, random, string
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from app.models import Progress
from app.serializers import ProgressSerializer

# Jars
import subprocess

# PyCuda
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Computing
import numpy as np
from math import sqrt
import tifffile
import psutil
from cpuinfo import get_cpu_info
import nvgpu

# Benchmark
import time

from io import BytesIO


class SystemUsage(viewsets.ViewSet):
    def list(self, request):
        try:
            gpu_info = nvgpu.gpu_info()[0]
            response = HttpResponse(json.dumps({
                'cpu_name': get_cpu_info()['brand_raw'],
                'cpu_count': psutil.cpu_count(),
                'ram_usage': dict(psutil.virtual_memory()._asdict()),
                'cpu_usage': psutil.cpu_percent(percpu = True),
                'gpu_name': gpu_info['type'],
                'gpu_usage': np.around(gpu_info['mem_used_percent'], decimals = 1)
            }))
        except:
            response = HttpResponse(json.dumps({
                'cpu_name': get_cpu_info()['brand_raw'],
                'cpu_count': psutil.cpu_count(),
                'ram_usage': dict(psutil.virtual_memory()._asdict()),
                'cpu_usage': psutil.cpu_percent(percpu = True),
            }))
        response["Content-Type"] = 'application/json'
        response.status_code = status.HTTP_200_OK
        return response

# def id_generator(size, chars=string.ascii_uppercase + string.digits):
#     return ''.join(random.choice(chars) for _ in range(size))

class ProgressViewSet(viewsets.ModelViewSet):
    queryset = Progress.objects.all()
    serializer_class = ProgressSerializer

class JarProcessing(viewsets.ViewSet):
    def create(self, request):
        # Get parameters from request data
        parameters = json.loads(request.data.get("processing_info"))
        filename = parameters.get("filename")
        method = parameters.get("method")
        predefinied_mask = request.data.get("mask")
        mask_filename = ''
        R = parameters.get("R")
        T = parameters.get("T")

        # Save image on disk and create output filename
        if not os.path.isfile(filename):
            default_storage.save(filename, ContentFile(request.data["image"].read()))
        output_filename = filename.split('.')[0] + '_r{}'.format(R) + '_t{}'.format(T) + '_' + method + '.tif'

        # Save progress data in database
        progress = Progress(name = output_filename, progress = 0)
        progress.save()

        process_args = [
            'java',
            '-jar',
            '/tiffProcessing/fastSDA.jar',
            '-file',
            'file={}'.format(filename),
            '-r',
            'r={}'.format(R),
            '-t',
            't={}'.format(T),
            '-method',
            'method={}'.format(method),
        ]

        # Custom mask from json
        if predefinied_mask:
            mask_filename = predefinied_mask.name.split('.')[0] + '_r{}'.format(R) + '_t{}'.format(T) + '_' + method + '.json'

            # Save mask and append to process arguments
            if not os.path.isfile(mask_filename):
                default_storage.save(mask_filename, predefinied_mask)

            process_args.append("-mask")
            process_args.append("mask={}".format(mask_filename))

        # Run multithreading java process
        process = subprocess.Popen(
            process_args,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )

        while True:
            # Read stdout and cast from bytes to string
            output = process.stdout.readline().strip()

            # Print stdout if not empty
            if output:
                if "Progress" in output:
                    index = output.find(" p=")
                    percent = 0
                    progress.progress = int(output[index+3:])
                    progress.save()

                # Thread has finished image processing
                if "Finished" in output:
                    # It is able to open that image and send back to the front
                    progress.delete()
            
            # Check if process has finished
            return_code = process.poll()

            # If return_code has not None value, process has finished
            if return_code is not None:
                break
            
            time.sleep(0.5)

        # Remove custom mask
        if predefinied_mask and os.path.isfile(mask_filename):
            os.remove(mask_filename)

        # Read binary data
        with open(output_filename, 'rb') as fp:
            data = fp.read()

        # Remove image and output image
        if os.path.isfile(filename):
            os.remove(filename)
        os.remove(output_filename)

        # Response
        response = FileResponse(
            io.BytesIO(data),
            as_attachment=True
        )
        response["Content-Type"] = 'multipart/form-data'
        response.status_code = status.HTTP_200_OK
        return response

class KernelProcessing(viewsets.ViewSet):
    def create(self, request):
        ctx = cuda.Device(0).make_context()

        # Kernele
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
        __global__ void gpu_float_mask_sda(const int X, const int Y, const int R, const int T, const int maskLength, const float *mask, const int *m, int *result)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza macierzy m i result
            int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny macierzy m i result
            // X * Y * pageNumber
            int pageId = X * Y * blockIdx.z;

            int x, y, dx, ry, from, to, lxbound, rxbound;
            float p, c;
            if ((row < X) && (col < Y)) {
                lxbound = MAX(0, row - R);
                rxbound = MIN(X, row + R + 1);
                p = 0;
                c = 0;
                for(x = lxbound; x < rxbound; x++) {
                    dx = row - x;

                    // (getHalfRing, ry jako int)
                    ry = getHalfRing(R, dx);

                    from = MAX(0, col - ry + 1);
                    to = MIN(Y, col + ry);
                    for(y = from; y < to; y++) {
                        // Przesuniecie bitowe
                        float maskValue = mask[(int) (my_sqrt((x - row) * (x - row) + (y - col) * (y - col)) / R * maskLength)];
                        p += maskValue;
                        c += ((((m[pageId + (row * Y + col)] - T) - m[pageId + (x * Y + y)]) >> 31) * maskValue);
                    }
                }
                result[pageId + (row * Y + col)] = 256 + (256 * c) / p;
            }
        }
        __global__ void gpu_int_mask_sda(const int X, const int Y, const int R, const int T, const int maskLength, const int *mask, const int *m, int *result)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza macierzy m i result
            int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny macierzy m i result
            // X * Y * pageNumber
            int pageId = X * Y * blockIdx.z;

            int x, y, dx, ry, drx, dry, from, to, lxbound, rxbound;
            float p, c;
            if ((row < X) && (col < Y)) {
                lxbound = MAX(0, row - R);
                rxbound = MIN(X, row + R + 1);
                p = 0;
                c = 0;
                for(x = lxbound; x < rxbound; x++) {
                    dx = row - x;
                    drx = dx + R;

                    // (getHalfRing, ry jako int)
                    ry = getHalfRing(R, dx);

                    from = MAX(0, col - ry + 1);
                    to = MIN(Y, col + ry);
                    for(y = from; y < to; y++) {
                        dry = y + R - col;

                        // If
                        //if ((x - row) * (x - row) + (y - col) * (y - col) < R * R) {
                        //    int maskValue = mask[drx * (2*R+1) + dry];
                        //    p += maskValue;
                        //    if (m[pageId + (x * Y + y)] + T <= m[pageId + (row * Y + col)]) {
                        //        c += maskValue;
                        //    }
                        //}

                        // Przesuniecie bitowe
                        p += mask[drx * (2*R+1) + dry];
                        c += ((((m[pageId + (row * Y + col)] - T) - m[pageId + (x * Y + y)]) >> 31) * mask[drx * (2*R+1) + dry]);

                        // Przepisanie obrazka
                        // result[row * Y + col] = m[row * Y + col];
                    }
                }
                result[pageId + (row * Y + col)] = 256 + (256 * c) / p;
            }
        }
        """)

        # Get parameters from request data
        parameters = json.loads(request.data.get("processing_info"))
        filename = parameters.get("filename")
        method = parameters.get("method")
        predefinied_mask = request.data.get("mask")
        pages = parameters.get("pages")
        X = int(parameters.get("X"))
        Y = int(parameters.get("Y"))
        R = int(parameters.get("R"))
        T = int(parameters.get("T"))
    
        # Save image and load to tifffile, then remove from disk
        if not os.path.isfile(filename):
            default_storage.save(filename, ContentFile(request.data["image"].read()))
        img = tifffile.imread(filename)
        if os.path.isfile(filename):
            os.remove(filename)

        # Mask / Custom Mask and result array
        if not predefinied_mask:
            gpu_sda = mod.get_function("gpu_int_mask_sda")
            mask = [[1 if (R - i)*(R - i) + (R - j)*(R - j) <= R * R else 0 for i in range(2 * R + 1)] for j in range(2 * R + 1)]
            mask = np.array(mask, dtype=np.int32)
        else:
            gpu_sda = mod.get_function("gpu_float_mask_sda")
            mask = json.loads(predefinied_mask.read().decode("utf-8"))["mask"]
            mask = np.array(mask, dtype=np.float32)

        # Benchmark
        start = cuda.Event()
        end = cuda.Event()
        time = 0

        # GPU data
        start.record()
        m_gpu = gpuarray.to_gpu(img.astype(np.int32))
        mask_gpu = gpuarray.to_gpu(mask)
        result_gpu = gpuarray.empty((pages, Y, X), dtype=np.int32)

        # Kernel call
        gpu_sda(
            np.int32(X),
            np.int32(Y),
            np.int32(R),
            np.int32(T),
            np.int32(mask.shape[0]),  # Used only for predefinied mask
            mask_gpu,
            m_gpu,
            result_gpu,
            block=(32, 32, 1),
            grid=((X+31)//32, (X+31)//32, pages)
        )

        # GPU results
        result_gpu_kernel = result_gpu.get()
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        time += secs

        # Save processed image
        output_filename = filename.split('.')[0] + '_r{}'.format(R) + '_t{}'.format(T) + '_' + method + '.tif'
        for i in range(pages):
            tifffile.imwrite(output_filename, result_gpu_kernel[i].astype(img.dtype), append=True, compression='deflate')

        # Finish life of gpu objects - free memory
        ctx.pop()
        del mask_gpu, m_gpu, result_gpu, mod, gpu_sda, start, end, ctx

        # Read binary data and remove
        with open(output_filename, 'rb') as fp:
            data = fp.read()
        if os.path.isfile(output_filename):
            os.remove(output_filename)
        print("time: ", time)

        # Response
        response = FileResponse(
            io.BytesIO(data),
            as_attachment=True
        )
        response["Content-Type"] = 'multipart/form-data'
        response.status_code = status.HTTP_200_OK
        return response
