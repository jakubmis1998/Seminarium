from django.contrib.auth.models import User
from rest_framework import viewsets, permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action, api_view
from django.http import FileResponse, HttpResponse
import io, os, json, random, string
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
 
from .serializers import UserSerializer, PersonSerializer
from .models import Person

# PyCuda
import pycuda.driver as cuda
import pycuda.autoinit
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

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

cuda.init()
device = cuda.Device(0)

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
 
class PersonViewSet(viewsets.ModelViewSet):
    queryset = Person.objects.all()
    serializer_class = PersonSerializer
    ordering_fields = '__all__'
 
@api_view(['GET', 'POST'])
def example(request):
    if request.method == 'POST':
        print("POST")
        A = np.array(request.data["a"], dtype=np.int32)
        B = np.array(request.data["b"], dtype=np.int32)
        return Response({'result': A+B}, status=status.HTTP_200_OK)
    elif request.method == 'GET':
        print("GET")
        return Response({'result': 2}, status=status.HTTP_200_OK)


@api_view(['POST'])
def read_and_return(request):
    print("Reading TIFF file")
    response = FileResponse(
        io.BytesIO(request.data["image"].read()),
        as_attachment=True
    )

    response["Content-Type"] = 'multipart/form-data'
    response.status_code = status.HTTP_200_OK
    print("Return response with file\n\n")
    return response


@api_view(['POST'])
def image_processing(request):
    if request.method == 'POST':

        print("Reading TIFF file")
        path = default_storage.save("plik.tif", ContentFile(request.data["image"].read()))
        print("Saved on disk")

        img = tifffile.imread('plik.tif')
        os.remove('plik.tif')
        print("Removed from disk")

        response = HttpResponse(json.dumps({'shape': img.shape}))
        response["Content-Type"] = 'application/json'
        response.status_code = status.HTTP_200_OK
        print("Return shape\n\n")

        return response

@api_view(['GET'])
def system_usage(request):
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

@api_view(['POST'])
def kernel_processing(request):
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
    filename = parameters.get("filename")
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

    # # Wynik GPU
    result_gpu_kernel = result_gpu.get()
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    time += secs

    tmp_name = f"{id_generator(15)}.tif"
    # Zapisanie przerobionego pliku na dysk
    for i in range(pages):
        tifffile.imwrite(tmp_name, result_gpu_kernel[i].astype(img.dtype), append=True, compression='deflate')

    ctx.pop()

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
