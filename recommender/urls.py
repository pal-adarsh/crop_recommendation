from django.urls import path

from . import views

app_name = "recommender"

urlpatterns = [
    path("", views.home, name="home"),
    path("models/", views.model_comparison, name="model_comparison"),
    path("insights/", views.insights, name="insights"),
]
