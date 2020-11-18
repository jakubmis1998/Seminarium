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
 
import numpy as np
import tifffile
import psutil

# GPU
# import pycuda.driver as cuda
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# from pycuda.compiler import SourceModule

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
    print(request.data)

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
