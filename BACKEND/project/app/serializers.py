from rest_framework import serializers
from app.models import Progress


class ProgressSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Progress
        fields = ['id', 'name', 'progress']