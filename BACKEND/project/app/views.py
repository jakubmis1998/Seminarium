from django.contrib.auth.models import User
from rest_framework import viewsets, permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action, api_view
from django.http import FileResponse
import io
 
from .serializers import UserSerializer, PersonSerializer
from .models import Person
 
import numpy as np
import tifffile

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
def calculate(request):
    if request.method == 'POST':
        print("POST")
        A = np.array(request.data["a"], dtype=np.int32)
        B = np.array(request.data["b"], dtype=np.int32)
        return Response({'result': A+B}, status=status.HTTP_200_OK)
    elif request.method == 'GET':
        print("GET")
        return Response({'result': 2}, status=status.HTTP_200_OK)

@api_view(['POST'])
def image_processing(request):
    if request.method == 'POST':
        print("DOSTALEM")
        print(request.data)
        # tifffile.imwrite('tmp.tif', request.data['image'])
        # img = tifffile.imread('tmp.tif')
        # print(img.shape)
        response = FileResponse(
            io.BytesIO(request.data['image'].read()),
            as_attachment=True,
            filename="somefilename.tif"
        )
        response["Content-Type"] = 'multipart/form-data'

        return response

        # return Response({'result': 1}, status=status.HTTP_200_OK)
