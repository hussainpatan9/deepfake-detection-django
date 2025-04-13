from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.conf import settings
from detector.utils import ModelSingleton
import os
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Refresh and reload all AI models in the system'

    def handle(self, *args, **kwargs):
        try:
            # Step 1: Clear model cache
            self.stdout.write('Clearing model cache...')
            for model_config in settings.AVAILABLE_MODELS:
                ModelSingleton.clear_model(model_config['file_path'])
            self.stdout.write(self.style.SUCCESS('✓ Model cache cleared'))

            # Step 2: Verify model files exist
            self.stdout.write('Verifying model files...')
            missing_models = []
            for model_config in settings.AVAILABLE_MODELS:
                if not os.path.exists(model_config['file_path']):
                    missing_models.append(model_config['name'])
            
            if missing_models:
                self.stdout.write(self.style.ERROR(
                    'Missing model files:\n' + '\n'.join(
                        f"- {name}" for name in missing_models
                    )
                ))
                return

            self.stdout.write(self.style.SUCCESS('✓ All model files verified'))

            # Step 3: Update database entries
            self.stdout.write('Updating model database entries...')
            call_command('load_models', verbosity=0)
            self.stdout.write(self.style.SUCCESS('✓ Database updated'))

            # Step 4: Test loading each model
            self.stdout.write('Testing models...')
            for model_config in settings.AVAILABLE_MODELS:
                try:
                    ModelSingleton.get_model(model_config['file_path'])
                    self.stdout.write(self.style.SUCCESS(
                        f"✓ {model_config['name']} loaded successfully"
                    ))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f"× Error loading {model_config['name']}: {str(e)}"
                    ))

            self.stdout.write(self.style.SUCCESS('\nModel refresh completed successfully!'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during refresh: {str(e)}'))
            logger.error(f'Model refresh failed: {str(e)}')
