from django.urls import path
from .views import AskView, AddDocView

urlpatterns = [
    path('ask/', AskView.as_view(), name='ask'),
    path('add/', AddDocView.as_view(), name='add'),
]