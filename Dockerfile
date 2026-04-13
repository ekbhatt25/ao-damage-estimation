FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (split for better layer caching)
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart \
    pillow numpy opencv-python-headless pycocotools \
    huggingface_hub ultralytics scikit-learn joblib \
    google-genai python-dotenv

# PyTorch CPU — HF Spaces free tier has no GPU
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy backend source
COPY backend/ ./backend/
COPY download_models.py ./download_models.py

# Download model weights from HF Hub at build time
# Increment CACHE_BUST to force re-download of models
ARG CACHE_BUST=2
RUN python download_models.py

# HF Spaces requires port 7860
EXPOSE 7860

WORKDIR /app/backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
