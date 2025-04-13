from django.core.management.base import BaseCommand
from django.conf import settings
from detector.models import DetectionModel
from django.db.utils import OperationalError
from django.db import connection

class Command(BaseCommand):
    help = 'Load available detection models into database'

    def handle(self, *args, **kwargs):
        try:
            # First, deactivate all existing models
            DetectionModel.objects.all().delete()
            self.stdout.write('Cleared existing models from database')

            # Add new models from settings
            for model_config in settings.AVAILABLE_MODELS:
                DetectionModel.objects.create(
                    name=model_config['name'],
                    file_path=model_config['file_path'],
                    description=model_config['description'],
                    architecture=model_config['architecture'],
                    input_size=model_config['input_size'],
                    is_active=True
                )
                self.stdout.write(self.style.SUCCESS(f"Added model: {model_config['name']}"))

            self.stdout.write(self.style.SUCCESS(f"Successfully loaded {len(settings.AVAILABLE_MODELS)} models"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading models: {str(e)}"))
