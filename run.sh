#!/bin/bash

# Build and run the ONNX2C web converter

echo "Building ONNX2C Web Converter Docker image..."

# Build the Docker image
docker build -t onnx2c-web-converter . -f web_converter/Dockerfile

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "Starting the web application..."
    echo "The application will be available at http://localhost:5000"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    # Run the container
    docker run -p 5000:5000 \
               --name onnx2c-converter \
               --rm \
               -v "$(pwd)/web_converter/app/uploads:/app/web_app/uploads" \
               -v "$(pwd)/web_converter/app/generated:/app/web_app/generated" \
               onnx2c-web-converter
else
    echo "❌ Failed to build Docker image"
    exit 1
fi