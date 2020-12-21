# backend/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf.urls import include, url
from rest_framework import routers
from SPI import views

router = routers.DefaultRouter()
router.register(r'todos', views.TodoView, 'todo')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('SPI.urls')),
    url('api/', include(router.urls))
]