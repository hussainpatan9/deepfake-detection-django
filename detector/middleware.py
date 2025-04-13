from django.http import HttpResponseForbidden
from django.conf import settings

class SecurityMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == 'POST' and request.FILES:
            for file in request.FILES.values():
                # Check file size
                if file.size > settings.FILE_UPLOAD_MAX_MEMORY_SIZE:
                    return HttpResponseForbidden("File too large. Maximum size is 10MB.")
                
                # Check file type
                if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
                    return HttpResponseForbidden("Invalid file type. Only JPG, JPEG, and PNG are allowed.")

        response = self.get_response(request)
        return response
