from django.urls import include, path
from rest_framework import routers
import api_practice.api as api
import api_practice.api2 as api2

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('warmer', api.warmer),
    path('cooler', api.cooler),
    path('brighten', api.brighten),
    path('darken', api.darken),
    path('oldify', api.oldify),
    path('ig_filter', api.ig_filter),
    path('face_average', api2.face_average),
    path('face_average2', api2.face_average2),
    path('face_morph', api2.face_morph),
    path('video_morph', api2.video_morph),
    path('face_swap', api2.face_swap),
    path('video_average', api2.video_average),
    path('test', api2.test),
    path('sketch', api2.sketch),
]
