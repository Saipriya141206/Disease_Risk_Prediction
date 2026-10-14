from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('assessment/', views.assessment, name='assessment'),
    path('signup/', views.sign_up, name='signup'),
    path('login/', views.sign_in, name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]