FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better layer caching
COPY requirements.txt /app/

# Install dependencies with optimized settings
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ai.py /app/
COPY rp_handler.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start the container
CMD ["python", "-u", "rp_handler.py"] 