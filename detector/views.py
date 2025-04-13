import platform
import django
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib import messages
from django.core.cache import cache
from django.conf import settings
from django.utils import timezone
from .forms import ImageUploadForm
from .utils import preprocess_image, ModelSingleton, predict_with_confidence, generate_gradcam_image  # Add this import
from .models import DetectionModel  # Add this at the top with other imports
import os
import logging
import numpy as np
from PIL import Image
import PIL.ExifTags
import time
import tensorflow as tf
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def home(request):
    if request.method == 'POST' and request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        messages.error(request, 'Please enable JavaScript for image analysis.')
        return redirect('detector:home')  # Add namespace
    form = ImageUploadForm()
    models = DetectionModel.objects.filter(is_active=True)
    return render(request, 'detector/home.html', {'form': form, 'models': models})

@require_http_methods(["GET", "POST"])  # Changed to allow GET
def analyze_image(request):
    if request.method == 'GET':
        # Redirect GET requests to result page
        return redirect('detector:result' + '?' + request.GET.urlencode())
        
    try:
        if 'image' not in request.FILES:
            raise ValidationError("No image file provided")

        form = ImageUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            raise ValidationError("Invalid form submission")

        image = request.FILES['image']
        selected_model = form.cleaned_data['model']
        
        # Validate file
        if image.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise ValidationError("Invalid file type. Only JPG, JPEG, and PNG are allowed.")
        if image.size > settings.FILE_UPLOAD_MAX_MEMORY_SIZE:
            raise ValidationError("File too large. Maximum size is 10MB.")

        # Process image and get prediction
        fs = FileSystemStorage()
        # Sanitize filename to avoid cache key issues
        safe_filename = "".join(c for c in image.name if c.isalnum() or c in ('-', '_', '.'))
        filename = fs.save(safe_filename, image)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        uploaded_file_url = fs.url(filename)
        
        try:
            start_time = time.time()
            # Load and validate model
            model = ModelSingleton.get_model(selected_model.file_path)
            
            # Preprocess image
            processed_image = preprocess_image(image_path)
            
            # Make prediction
            prediction_result = predict_with_confidence(model, processed_image)
            
            processing_time = time.time() - start_time
            
            result = {
                'uploaded_file_url': uploaded_file_url,
                'prediction': 'Real' if prediction_result['is_real'] else 'Fake',
                'confidence': f"{prediction_result['confidence'] * 100:.1f}%",
                'confidence_level': prediction_result['confidence_level'],
                'raw_probability': prediction_result['raw_probability'],
                'filename': filename,
                'analyzed_at': timezone.now().isoformat(),
                'processing_time': f"{processing_time:.2f}",
                'model_name': selected_model.name,
                'model_architecture': selected_model.architecture,
                'model_id': selected_model.id,  # Add model ID to result
            }
            
            # Cache the result
            cache_key = f'analysis_{filename}'
            cache.set(cache_key, result, 3600)
            cache.set(cache_key, result, 3600)
            
            logger.info(f"Image {filename} analyzed: {result['prediction']} ({result['confidence']}) - {result['confidence_level']} confidence")
            
            # Don't cleanup the image immediately in development
            if not settings.DEBUG:
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup image {filename}: {str(e)}")
            
            return JsonResponse(result)
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(image_path):
                os.remove(image_path)
            raise e
                
    except ValidationError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JsonResponse({'error': 'An error occurred processing the image'}, status=500)

@require_http_methods(["GET"])
def test_model(request):
    """Test endpoint to verify model functionality"""
    try:
        model_id = request.GET.get('model_id')
        if not model_id:
            # Use first available model as default
            model = DetectionModel.objects.filter(is_active=True).first()
        else:
            model = DetectionModel.objects.get(id=model_id)

        # Load model
        model_instance = ModelSingleton.get_model(model.file_path)
        
        # Create test tensor
        test_tensor = np.random.random((1, 256, 256, 3)).astype(np.float32)
        
        # Make test prediction
        prediction = model_instance.predict(test_tensor, verbose=0)
        
        result = {
            'status': 'success',
            'message': 'Model test successful',
            'model_name': model.name,
            'model_loaded': model_instance is not None,
            'test_prediction': float(prediction[0][0]),
            'timestamp': timezone.now().isoformat()
        }
        
        logger.info(f"Model test successful for {model.name}")
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def result(request):
    try:
        # Parse ISO format datetime
        analyzed_at = request.GET.get('analyzed_at')
        if analyzed_at:
            analyzed_at = datetime.fromisoformat(analyzed_at).astimezone(pytz.UTC)
        
        # Get model input size from settings or use default
        input_size = getattr(settings, 'MODEL_INPUT_SIZE', (256, 256))
        
        context = {
            'uploaded_file_url': request.GET.get('uploaded_file_url'),
            'prediction': request.GET.get('prediction', 'Unknown'),
            'confidence': request.GET.get('confidence', '0%'),
            'confidence_level': request.GET.get('confidence_level', 'unknown'),
            'raw_probability': float(request.GET.get('raw_probability', 0)),
            'analyzed_at': analyzed_at,
            'analyzed_at_local': analyzed_at.astimezone() if analyzed_at else None,
            'processing_time': float(request.GET.get('processing_time', '0.00')),
            'filename': request.GET.get('filename', ''),
            'model_name': request.GET.get('model_name', 'Unknown Model'),
            'model_architecture': request.GET.get('model_architecture', 'Unknown Architecture'),
            'device_used': 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU',
            'model_id': request.GET.get('model_id'),  # Add model ID to context
            'model_info': {
                'name': request.GET.get('model_name', 'Unknown Model'),
                'architecture': request.GET.get('model_architecture', 'Unknown Architecture'),
                'input_size': f"{input_size[0]}x{input_size[1]}",
                'framework': f"TensorFlow {tf.__version__}"
            }
        }

        # Get image metadata if available
        image_path = os.path.join(settings.MEDIA_ROOT, context['filename'])
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                context['original_size'] = f"{img.size[0]} x {img.size[1]}"
                context['image_type'] = img.format
                
                # Extract EXIF data
                try:
                    exif = {
                        PIL.ExifTags.TAGS[k]: v
                        for k, v in img._getexif().items()
                        if k in PIL.ExifTags.TAGS
                    } if img._getexif() else {}
                    context['metadata'] = {k: str(v) for k, v in exif.items()}
                except:
                    context['metadata'] = None

        if not context['uploaded_file_url']:
            messages.error(request, 'No image was provided for analysis')
            return redirect('detector:home')
            
        # Cleanup old files in the background
        if settings.CLEANUP_FILES_IN_DEVELOPMENT or not settings.DEBUG:
            try:
                import threading
                threading.Thread(target=cleanup_old_files, 
                              args=(settings.MEDIA_ROOT,), 
                              daemon=True).start()
            except Exception as e:
                logger.warning(f"Failed to start cleanup thread: {e}")


        return render(request, 'detector/result.html', context)
        
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        messages.error(request, 'Error displaying results')
        return redirect('detector:home')

@require_http_methods(["GET"])
def about(request):
    # Check if profile image exists
    profile_image_path = os.path.join(settings.STATIC_ROOT, 'detector', 'images', 'profile.jpg')
    profile_image_exists = os.path.exists(profile_image_path)
    
    models = DetectionModel.objects.filter(is_active=True)
    context = {
        'models': models,
        'total_models': models.count(),
        'device_used': 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU',
        'tensorflow_version': tf.__version__,
        'python_version': platform.python_version(),
        'django_version': django.get_version(),
        'profile_image_exists': profile_image_exists,
    }
    return render(request, 'detector/about.html', context)

@require_http_methods(["GET"])
def generate_gradcam(request):
    try:
        image_path = os.path.join(settings.MEDIA_ROOT, request.GET.get('filename'))
        model_id = request.GET.get('model_id')
        
        if not os.path.exists(image_path):
            raise ValidationError("Image not found")
            
        model = DetectionModel.objects.get(id=model_id)
        model_instance = ModelSingleton.get_model(model.file_path)
        
        # Ensure the model is called with a dummy input to initialize it
        dummy_input = np.random.random((1, 256, 256, 3)).astype(np.float32)
        _ = model_instance(dummy_input)
        
        # model.get_layer("conv5_block32_concat")
# model.get_layer("conv5_block3_out")

        # Update target layers for each architecture
        target_layer = {
            # 'DenseNet121': 'conv5_block16_2_conv',
            'DenseNet121': 'conv5_block32_concat',

            # 'ResNet50': 'conv5_block3_3_conv',
            'ResNet50': 'conv5_block3_out',

            'Custom CNN': 'conv2d_5'  # Updated to use the last conv layer of your CNN
        }.get(model.architecture)
        
        if not target_layer:
            logger.warning(f"Using fallback layer for architecture: {model.architecture}")
            # For Custom CNN, try to find the last convolutional layer
            for layer in reversed(model_instance.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer = layer.name
                    logger.info(f"Found conv layer for Custom CNN: {target_layer}")
                    break
        
        if not target_layer:
            raise ValueError(f"Could not find suitable conv layer for architecture: {model.architecture}")
            
        logger.info(f"Using target layer '{target_layer}' for {model.architecture}")
        gradcam_image = generate_gradcam_image(model_instance, image_path, target_layer)
        
        return JsonResponse({
            'gradcam_image': gradcam_image
        })
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

