from rest_framework import serializers
from .models import Cloud_ramanathapuram

class Cloud_ramanathapuramSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cloud_ramanathapuram
        fields = '__all__'
