from django.db import models

class Detection(models.Model):
    image = models.ImageField(upload_to='uploads/')
    prediction = models.CharField(max_length=10)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction} ({self.confidence:.2f}%)"

class DetectionModel(models.Model):
    name = models.CharField(max_length=100)
    file_path = models.CharField(max_length=255)
    description = models.TextField()
    architecture = models.CharField(max_length=100)
    input_size = models.CharField(max_length=50)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.architecture})"
