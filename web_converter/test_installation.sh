#!/bin/bash

echo "🧪 Testing ONNX2C Web Converter Installation"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker is installed"

# Check if the project files exist
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found. Make sure you're in the web_converter directory."
    exit 1
fi

echo "✅ Project files found"

# Try to build the Docker image
echo "🔨 Building Docker image (this may take a few minutes)..."
if docker build -t onnx2c-web-test . > /dev/null 2>&1; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image. Check the logs above."
    exit 1
fi

# Test run the container briefly
echo "🚀 Testing container startup..."
CONTAINER_ID=$(docker run -d -p 5001:5000 onnx2c-web-test)

# Wait a moment for startup
sleep 5

# Check if container is running
if docker ps | grep -q $CONTAINER_ID; then
    echo "✅ Container started successfully"
    echo "🌐 Test server would be available at http://localhost:5001"
else
    echo "❌ Container failed to start"
    docker logs $CONTAINER_ID
    exit 1
fi

# Clean up
docker stop $CONTAINER_ID > /dev/null 2>&1
docker rm $CONTAINER_ID > /dev/null 2>&1

echo ""
echo "🎉 All tests passed! The ONNX2C Web Converter is ready to use."
echo ""
echo "To start the application:"
echo "  Linux/macOS: ./run.sh"
echo "  Windows:     run.bat"
echo "  Docker Compose: docker-compose up --build"
echo ""
echo "The application will be available at http://localhost:5000"