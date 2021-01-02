from django.urls import include, path
from rest_framework import routers
from django.contrib import admin
from app import views

router = routers.DefaultRouter()
router.register(r'jar_processing', views.JarProcessing, basename="jar_processing")
router.register(r'kernel_processing', views.KernelProcessing, basename="kernel_processing")
router.register(r'system_usage', views.SystemUsage, basename="system_usage")
router.register(r'processing_progress', views.ProgressViewSet, basename="processing_progress")


urlpatterns = [
    path('', include(router.urls)),
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
