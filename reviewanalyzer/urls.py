from django.urls import path
from .views import ReviewView

urlpatterns = [
    path('status/', ReviewView.as_view(), name = 'spam_classification'),
]