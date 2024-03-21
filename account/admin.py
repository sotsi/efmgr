from django.contrib import admin
from .models import PVString, DayIrradiance

@admin.register(PVString)
class PVStringAdmin(admin.ModelAdmin):
	list_display = ['Username','Azimuth','Pnom']
	prepopulated_fields = {'slug': ('Username','Azimuth','Pnom')}
    
@admin.register(DayIrradiance)
class DayIrradianceAdmin(admin.ModelAdmin):
	list_display = ['DateNow','IrrClass']
	prepopulated_fields = {'slug': ('DateNow','IrrClass')}
