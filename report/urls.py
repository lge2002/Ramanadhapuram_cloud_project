
from django.urls import path
from . import views

app_name = 'report'

urlpatterns = [
    path('report/', views.report_view, name='report'),
    path('download-report/', views.download_report_pdf, name='download_report_pdf'),
 
]