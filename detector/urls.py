from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/', views.analyze_image, name='analyze'),
    path('result/', views.result, name='result'),
    path('test-model/', views.test_model, name='test-model'),
    path('about/', views.about, name='about'),  # Add this line
    path('gradcam/', views.generate_gradcam, name='gradcam'),
]
