from django.urls import path
from .views import CloudAPIView

urlpatterns = [
    path('api/cloud/', CloudAPIView.as_view(), name='cloud-api'),
]
