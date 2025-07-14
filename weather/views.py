from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Cloud_ramanathapuram
from .serializers import Cloud_ramanathapuramSerializer

class CloudAPIView(APIView):
    def get(self, request):
        data = Cloud_ramanathapuram.objects.all().order_by('-timestamp')
        serializer = Cloud_ramanathapuramSerializer(data, many=True)
        return Response(serializer.data)