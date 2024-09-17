# Dockerfile
# Use an official Python 3.11 runtime as a parent image
FROM python:3.11-slim

# Define environment variable
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN TMPDIR='/tmp' pip install -r requirements.txt --cache-dir=/tmp/

# Make port 5001 available to the world outside this container
EXPOSE 5001
