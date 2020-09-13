from django.urls import path, include
from . import views

urlpatterns = [
	path('', views.index, name='index'),
	path('detect/', views.videoFeed, name='detect')
]