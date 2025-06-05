from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('predict_frame/', views.predict_frame, name='predict_frame'),
]
