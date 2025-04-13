import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import tensorflow as tf
from django.apps import AppConfig
from django.conf import settings

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        # Prevent running twice in development
        if os.environ.get('RUN_MAIN') != 'true':
            return

        if hasattr(self, 'debug') and self.debug:
            tf.get_logger().setLevel('ERROR')
            tf.autograph.set_verbosity(0)
