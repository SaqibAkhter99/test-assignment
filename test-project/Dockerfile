# Dockerfile
# Use a base image that includes Python. A slim version is efficient.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system libraries required by OpenCV or Pillow for image processing
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY test-project/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the working directory
COPY . .

# Cerebrium platform will handle the CMD instruction to run the application.

