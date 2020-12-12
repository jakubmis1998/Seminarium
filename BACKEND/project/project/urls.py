from django.urls import include, path
from rest_framework import routers
from django.contrib import admin
from app import views

router = routers.DefaultRouter()
router.register(r'users/', views.UserViewSet)
router.register(r'people/', views.PersonViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('example/', views.example),
    path('read_and_return/', views.read_and_return),
    path('image_processing/', views.image_processing),
    path('system_usage/', views.system_usage),
    path('kernel/', views.kernel),
]
