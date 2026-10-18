FROM python:3.10-slim

# Create user with UID 1000 (Hugging Face standard)
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install system dependencies for OpenCV & YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . .

# Hugging Face Spaces port
EXPOSE 7860

CMD ["python", "app.py"]
