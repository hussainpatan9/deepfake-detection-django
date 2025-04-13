document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const fileInput = document.querySelector('input[type="file"]');
    const preview = document.getElementById('preview');
    const previewContainer = document.querySelector('.preview-container');
    const analyzeBtn = document.getElementById('analyze-btn');
    const spinner = analyzeBtn.querySelector('.spinner-border');
    const dropZone = document.querySelector('.dropzone-area');
    const progressBar = document.querySelector('.progress');
    const progressBarInner = progressBar.querySelector('.progress-bar');
    const removeImageBtn = document.getElementById('remove-image');
    const themeToggle = document.getElementById('theme-toggle');
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                displayPreview(file);
                analyzeBtn.disabled = false;
            }
        }
    }

    fileInput?.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && validateFile(file)) {
            displayPreview(file);
            analyzeBtn.disabled = false;
        }
    });

    function validateFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!allowedTypes.includes(file.type)) {
            alert('Please upload only JPG or PNG images.');
            return false;
        }

        if (file.size > maxSize) {
            alert('File size must be less than 10MB.');
            return false;
        }

        return true;
    }

    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';
            previewContainer.classList.add('fade-in');
        }
        reader.readAsDataURL(file);
    }

    removeImageBtn?.addEventListener('click', function() {
        fileInput.value = '';
        previewContainer.style.display = 'none';
        analyzeBtn.disabled = true;
    });

    form?.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!fileInput.files.length) {
            alert('Please select an image first');
            return;
        }

        showLoading();
        analyzeBtn.disabled = true;
        spinner.classList.remove('d-none');
        progressBar.style.display = 'block';
        
        const formData = new FormData(form);
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                progressBarInner.style.width = percentComplete + '%';
            }
        });

        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4) {
                hideLoading();
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        if (response.error) {
                            throw new Error(response.error);
                        }
                        // Update to use result URL instead
                        window.location.href = `/result/?${new URLSearchParams(response)}`;
                    } catch (e) {
                        console.error('Error:', e);
                        alert(e.message || 'An error occurred during analysis');
                    }
                } else {
                    alert('An error occurred during analysis. Please try again.');
                }
                analyzeBtn.disabled = false;
                spinner.classList.add('d-none');
                progressBar.style.display = 'none';
            }
        };

        xhr.open('POST', form.action, true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.setRequestHeader('X-CSRFToken', document.querySelector('[name=csrfmiddlewaretoken]').value);
        xhr.send(formData);
    });

    // Dark mode toggle
    function setTheme(theme) {
        document.documentElement.setAttribute('data-bs-theme', theme);
        localStorage.setItem('theme', theme);
        themeToggle.innerHTML = theme === 'dark' ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }

    themeToggle?.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-bs-theme');
        setTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });

    // Initialize theme
    const savedTheme = localStorage.getItem('theme') || 
        (prefersDarkScheme.matches ? 'dark' : 'light');
    setTheme(savedTheme);

    // Show loading overlay
    function showLoading() {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay fade-in';
        overlay.innerHTML = '<div class="spinner"></div>';
        document.body.appendChild(overlay);
    }

    // Hide loading overlay
    function hideLoading() {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.classList.add('fade-out');
            setTimeout(() => overlay.remove(), 500);
        }
    }

    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
