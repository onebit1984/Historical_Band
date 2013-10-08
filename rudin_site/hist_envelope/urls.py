from django.conf.urls import patterns,url
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from hist_envelope import views

urlpatterns = patterns('',
        url(r'^$',views.index,name='index'),
        url(r'^(?P<input_day>\d+)/$',views.show_dates, name='show_dates'),
        url(r'^(?P<input_day>\d+)/show_weather/$',views.show_similar_weather, name='show_weather'),
        url(r'^(?P<input_day>\d+)/show_load/$',views.show_similar_weather_load, name='show_load'),
        url(r'^(?P<input_day>\d+)/show_envelope',views.show_similar_weather_envelope, name='show_envelope'),
        )

urlpatterns += staticfiles_urlpatterns()
