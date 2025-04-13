from django import forms
from .models import DetectionModel

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        required=True,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )
    model = forms.ModelChoiceField(
        queryset=DetectionModel.objects.filter(is_active=True),
        empty_label=None,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
