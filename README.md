# Deepfake Detector

## Overview

The **Deepfake Detector** is a Django-based web application designed to detect deepfake images using advanced machine learning models. It leverages pre-trained architectures like DenseNet201, ResNet50, and a custom lightweight CNN for robust and efficient deepfake classification.

## Features

- **Multiple Model Support**: Includes DenseNet201, ResNet50, and a custom CNN for deepfake detection.
- **File Uploads**: Supports uploading images in formats like JPEG and PNG for analysis.
- **Caching**: Implements file-based caching for improved performance.
- **Logging**: Provides detailed logging for debugging and monitoring.
- **Security**: Enforces strict security settings, including CSRF protection and secure cookies.
- **File Cleanup**: Automatically removes uploaded files after a configurable time period.

## Project Structure

```
deepfake_detector/
├── config/                 # Django project configuration
│   ├── settings.py         # Main settings file
│   ├── urls.py             # URL routing
│   ├── wsgi.py             # WSGI entry point
├── detector/               # Main application
│   ├── middleware/         # Custom middleware
│   ├── models/             # Machine learning models
│   ├── static/             # Static files (CSS, JS, etc.)
│   ├── templates/          # HTML templates
│   ├── views.py            # Application views
├── media/                  # Uploaded media files
├── staticfiles/            # Collected static files
├── logs/                   # Log files
└── manage.py               # Django management script
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Django 3.2 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deepfake_detector
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Apply migrations:
   ```bash
   python manage.py migrate
   ```

5. Collect static files:
   ```bash
   python manage.py collectstatic
   ```

6. Start the development server:
   ```bash
   python manage.py runserver
   ```

7. Access the application at `http://127.0.0.1:8000`.

## Configuration

### Settings

- **Models Directory**: Ensure the `models` directory exists and contains the required `.keras` files.
- **Logging**: Logs are stored in the `logs/debug.log` file.
- **Static and Media Files**: Static files are served from the `staticfiles` directory, and uploaded media files are stored in the `media/uploads` directory.

### Environment Variables

For production, set the following environment variables:
- `SECRET_KEY`: Replace the default secret key with a secure one.
- `DEBUG`: Set to `False` in production.
- `ALLOWED_HOSTS`: Add your domain or server IP.

## Usage

1. Upload an image through the web interface.
2. Select the desired model for analysis.
3. View the results, including whether the image is classified as real or fake.

## Available Models

1. **DenseNet201 Deepfake Detector**
   - Architecture: DenseNet201
   - Input Size: 256x256
   - Description: Optimized for deepfake detection using transfer learning.

2. **ResNet50 Image Classifier**
   - Architecture: ResNet50
   - Input Size: 256x256
   - Description: Known for its residual connections, adapted for deepfake detection.

3. **Custom Lightweight CNN**
   - Architecture: Custom CNN
   - Input Size: 256x256
   - Description: Lightweight architecture for efficient classification.

## Security

- CSRF protection is enabled.
- Cookies are secured with `CSRF_COOKIE_SECURE` and `SESSION_COOKIE_SECURE`.
- X-Frame options are set to `DENY`.

## Logging

- Logs errors to `logs/debug.log`.
- Outputs info-level logs to the console.

## File Cleanup

Uploaded files are automatically removed after 24 hours. This behavior can be configured in the `settings.py` file.

## License

This project is licensed under the MIT License. See the `LICENSE.txt` file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Contact

For questions or support, please contact [your-email@example.com].