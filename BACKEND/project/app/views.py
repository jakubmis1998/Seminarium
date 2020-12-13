from django.contrib.auth.models import User
from rest_framework import viewsets, permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action, api_view
from django.http import FileResponse, HttpResponse
import io, os, json
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

# Benchmark
import time

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
    response = HttpResponse(json.dumps({
        'cpu_count': psutil.cpu_count(),
        'ram_usage': dict(psutil.virtual_memory()._asdict()),
        'cpu_usage': psutil.cpu_percent(percpu = True)
    }))
    response["Content-Type"] = 'application/json'
    response.status_code = status.HTTP_200_OK
    print("RETURN SYSTEM USAGE")
    return response

@api_view(['POST'])
def kernel_processing(request):
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    # Kernel
    # Każdy wątek oblicza jeden element result z m - czyli wątków jest X * Y.
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
    __global__ void gpu_int_mask_multi_thread(const int X, const int Y, const int R, const int *mask, const int *m, int *result)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza macierzy m i result
        int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny macierzy m i result
        int x, y, dx, ry, drx, dry, from, to, lxbound, rxbound;
        if ((row < X) && (col < Y)) {
            lxbound = MAX(0, row - R);
            rxbound = MIN(X, row + R + 1);
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
                    result[row * Y + col] -= (((m[row * Y + col] - m[x * Y + y]) >> 31) * mask[drx * (2*R+1) + dry]);
                }
            }
        }
    }
    """)
    gpu_int_mask_multi_thread = mod.get_function("gpu_int_mask_multi_thread")

    # Get parameters from request data
    parameters = json.loads(request.data["parameters"])
    filename = parameters["filename"]
    method = parameters["method"]
    depth = int(parameters["depth"])
    X = int(parameters["X"])
    Y = int(parameters["Y"])

    # Save image and load to tifffile, then remove from disk
    path = default_storage.save(filename, ContentFile(request.data["image"].read()))
    img = tifffile.imread(filename)
    os.remove(filename)

    R = int(parameters["parameters"][0]["R"])
    T = int(parameters["parameters"][0]["T"])
    print(R, T, X, Y)

    if depth == 8:
        m_with_rgba_canals = np.zeros((Y, X, 4), dtype=np.int8)
    elif depth == 16:
        m_with_rgba_canals = np.zeros((Y, X, 4), dtype=np.int16)
    elif depth == 32:
        m_with_rgba_canals = np.zeros((Y, X, 4), dtype=np.int32)
    elif depth == 64:
        m_with_rgba_canals = np.zeros((Y, X, 4), dtype=np.int64)

    for i in range(Y):
        for j in range(X):
            # First page - [0]
            m_with_rgba_canals[i][j] = np.array([img[0][i][j], 0, 0, 255])

    mask = np.random.randint(2, size=(2*R + 1, 2*R + 1), dtype=np.int32)
    result = np.zeros((Y, X), dtype=np.int32)

    # Zmienne do pomiaru czasu na GPU
    start = cuda.Event()
    end = cuda.Event()

    # # Dane GPU
    start.record()
    m_gpu = gpuarray.to_gpu(m_with_rgba_canals)
    mask_gpu = gpuarray.to_gpu(mask)
    result_gpu = gpuarray.empty((Y, X), dtype=np.int32)

    # # Wywołanie kernel'a
    gpu_int_mask_multi_thread(
        np.int32(X),
        np.int32(Y),
        np.int32(R),
        mask_gpu,
        m_gpu,
        result_gpu,
        block=(32, 32, 1),
        grid=((X+31)//32, (X+31)//32, 1)
    )

    # # Wynik GPU
    result_gpu_kernel = result_gpu.get()
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("GPU: %.7f s" % secs)

    ctx.pop()

    # Zapisanie przerobionego pliku na dysk
    tifffile.imwrite('processed.tiff', result_gpu_kernel.astype('uint8'))
    # Odczytanie go w formie binarnej
    with open('processed.tiff', 'rb') as fp:
        data = fp.read()
    os.remove('processed.tiff')
    # Zwrocenie na front
    response = FileResponse(
        io.BytesIO(data),
        as_attachment=True
    )
    response["Content-Type"] = 'multipart/form-data'
    response.status_code = status.HTTP_200_OK
    return response
    # return Response({
        # "result": result_gpu_kernel,
        # "time": secs,
        # "shape": img.shape
    # }, status=status.HTTP_200_OK)