import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.conf import settings
import numpy as np
import logging
import os
import time
from django.core.cache import cache
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import io
import base64

__all__ = [
    'preprocess_image',
    'predict_with_confidence',
    'ModelSingleton',
    'test_model_prediction',
    'grad_cam',
    'generate_gradcam_image',
]

logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and preprocess image
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Validate preprocessed image
        if img_array.shape != (1, 256, 256, 3):
            raise ValueError(f"Invalid image shape after preprocessing: {img_array.shape}")
        if not (0 <= img_array.min() <= img_array.max() <= 1.0):
            raise ValueError("Image normalization failed")
            
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def test_model_prediction(model):
    """Test model prediction with a sample tensor"""
    try:
        # Create a test tensor
        test_tensor = np.random.random((1, 256, 256, 3)).astype(np.float32)
        
        # Test prediction
        pred = model.predict(test_tensor)
        
        # Validate prediction shape and range
        if not (pred.shape == (1, 1) and 0 <= pred[0][0] <= 1):
            raise ValueError("Model prediction validation failed")
            
        logger.info("Model prediction test passed successfully")
        return True
    except Exception as e:
        logger.error(f"Model prediction test failed: {str(e)}")
        return False

class ModelSingleton:
    _instances = {}

    @classmethod
    def get_model(cls, model_path):
        if model_path not in cls._instances:
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                
                # Load the model
                cls._instances[model_path] = tf.keras.models.load_model(model_path)
                
                # Validate model architecture
                expected_input_shape = (None, 256, 256, 3)
                if cls._instances[model_path].input_shape != expected_input_shape:
                    raise ValueError(f"Invalid model input shape. Expected {expected_input_shape}")
                
                # Test model prediction
                if not test_model_prediction(cls._instances[model_path]):
                    raise ValueError("Model prediction validation failed")
                
                logger.info(f"Model loaded and validated successfully: {model_path}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {str(e)}")
                raise
                
        return cls._instances[model_path]

    @classmethod
    def clear_model(cls, model_path):
        if model_path in cls._instances:
            del cls._instances[model_path]

def cleanup_old_files(directory, max_age_hours=24):
    """Clean up old uploaded files"""
    try:
        current_time = time.time()
        cleaned = 0
        for filename in os.listdir(directory):
            if filename.startswith('.'): continue
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_hours * 3600:
                    os.remove(filepath)
                    cleaned += 1
        logger.info(f"Cleaned {cleaned} old files from {directory}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

def predict_with_confidence(model, image):
    """Make prediction with confidence check"""
    try:
        # Make prediction with verbose=0 to suppress progress bar
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            prediction = model.predict(image, verbose=0, batch_size=1)
        
        # Validate prediction
        if prediction.shape != (1, 1):
            raise ValueError(f"Invalid prediction shape: {prediction.shape}")
            
        prob = float(prediction[0][0])
        if not (0 <= prob <= 1):
            raise ValueError(f"Invalid prediction probability: {prob}")
        
        # Calculate confidence with threshold adjustment
        threshold = 0.5
        is_real = prob > threshold
        confidence = prob if is_real else (1 - prob)
        
        # Add confidence level category
        confidence_level = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
        
        return {
            'is_real': is_real,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'raw_probability': prob
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def grad_cam(model, image, class_idx, target_layer_name):
    """Generate Grad-CAM heatmap"""
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(target_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def generate_gradcam_image(model, image_path, target_layer_name='conv5_block16_2_conv'):
    """Generate Grad-CAM visualization"""
    try:
        plt.switch_backend('Agg')
        logger.info(f"Generating Grad-CAM for layer: {target_layer_name}")
        
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Ensure the model is called with a dummy input to initialize it
        dummy_input = np.random.random((1, 256, 256, 3)).astype(np.float32)
        _ = model(dummy_input)
        
        # Create Grad-CAM model directly
        target_layer = model.get_layer(target_layer_name)
        grad_model = Model(
            inputs=[model.input],
            outputs=[target_layer.output, model.output]
        )
        
        # Get the conv outputs and predictions
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        # Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by importance
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Load and resize original image
        original_img = load_img(image_path)
        img_array = img_to_array(original_img)
        
        # Resize heatmap to match original image size
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis],
            (img_array.shape[0], img_array.shape[1])
        ).numpy()
        
        # Create visualization
        plt.figure(figsize=(10, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array / 255)
        plt.title('Original Image')
        plt.axis('off')
        
        # Heatmap overlay
        plt.subplot(1, 2, 2)
        plt.imshow(img_array / 255)
        plt.imshow(heatmap_resized[..., 0], cmap='jet', alpha=0.4)
        plt.title('Grad-CAM Visualization')
        plt.axis('off')
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close('all')
        
        return base64.b64encode(image_png).decode()
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        plt.close('all')
        raise
