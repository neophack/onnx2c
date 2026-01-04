# Use Ubuntu as base image
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    python3 \
    python3-pip \
    gcc \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

    # Copy requirements and install Python dependencies
COPY web_converter/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Copy source files and dependencies to /app
COPY src/ /app/src/
COPY onnx/ /app/onnx/
COPY CMakeLists.txt /app/
COPY scripts/ /app/scripts/
COPY examples/ /app/examples/
COPY test/ /app/test/
COPY args/ /app/args/
COPY aixlog/ /app/aixlog/
COPY benchmark/ /app/benchmark/
COPY protobuf/ /app/protobuf/
COPY cmake_timestamp/ /app/cmake_timestamp/
COPY CMakeLists.txt /app/

# Build onnx2c
RUN mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make onnx2c && \
    cp onnx2c /usr/bin/onnx2c && \
    make clean

# Copy web converter files
COPY web_converter/ /app/
COPY web_converter/app /app/web_app/

# Set environment variables
ENV ONNX2C_PATH=/usr/bin/onnx2c
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Change to web app directory
WORKDIR /app/web_app

# Start the web application
CMD ["python3", "app.py"]