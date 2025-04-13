from django.contrib import admin
from .models import Detection

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('prediction', 'confidence', 'created_at')
    list_filter = ('prediction', 'created_at')
    search_fields = ('prediction',)
    ordering = ('-created_at',)
    
# Register your models here.
