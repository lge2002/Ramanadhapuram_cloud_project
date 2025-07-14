from django.urls import path
from . import views

urlpatterns = [
    path('live/',views.data, name="live_datas")
]