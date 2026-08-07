#!/usr/bin/env bash
set -euo pipefail

# Build onnx2c as a WebAssembly module for static browser deployment.
# The script automatically installs the Emscripten SDK if it is not found,
# and builds a host-native protoc for generating the ONNX protobuf bindings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-wasm"
NATIVE_BUILD_DIR="${PROJECT_ROOT}/build-native-protoc"
OUTPUT_DIR="${PROJECT_ROOT}/web_converter/wasm"
EMSDK_DIR="${PROJECT_ROOT}/emsdk"

# absl is fetched by protobuf via FetchContent at configure time. We pin it to
# a fixed local checkout so that, once downloaded, WASM builds work fully offline.
ABSL_SOURCE_DIR="${PROJECT_ROOT}/third_party/abseil-cpp"
ABSL_GIT_TAG="20250512.1"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------
# 1. Locate or install the Emscripten SDK
# ---------------------------------------------------------------------
if ! command -v emcc &> /dev/null; then
    echo "[build_wasm] Emscripten SDK not found in PATH."
    echo "[build_wasm] Checking local SDK at ${EMSDK_DIR}..."

    if [ ! -f "${EMSDK_DIR}/emsdk" ]; then
        echo "[build_wasm] Cloning Emscripten SDK..."
        git clone https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}"
    fi

    cd "${EMSDK_DIR}"

    echo "[build_wasm] Installing latest Emscripten SDK..."
    ./emsdk install latest

    echo "[build_wasm] Activating latest Emscripten SDK..."
    ./emsdk activate latest

    echo "[build_wasm] Setting up Emscripten environment..."
    # shellcheck source=/dev/null
    source ./emsdk_env.sh
fi

if ! command -v emcc &> /dev/null; then
    echo "[build_wasm] ERROR: emcc still not available after SDK setup."
    exit 1
fi

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------
# 2. Ensure Abseil (absl) is available locally for offline builds
# ---------------------------------------------------------------------
# protobuf pulls absl via FetchContent during cmake configure. By pointing
# FETCHCONTENT_SOURCE_DIR_ABSL at a persistent checkout we download once and
# can rebuild without a network connection afterwards.
if [ ! -f "${ABSL_SOURCE_DIR}/CMakeLists.txt" ]; then
    echo "[build_wasm] Abseil not found locally, downloading (tag ${ABSL_GIT_TAG})..."
    rm -rf "${ABSL_SOURCE_DIR}"
    git clone --depth 1 --branch "${ABSL_GIT_TAG}" https://github.com/abseil/abseil-cpp.git "${ABSL_SOURCE_DIR}"
else
    echo "[build_wasm] Using local Abseil at ${ABSL_SOURCE_DIR} (offline-ready)."
fi

ABSL_FETCH_FLAG="-DFETCHCONTENT_SOURCE_DIR_ABSL=${ABSL_SOURCE_DIR}"

# ---------------------------------------------------------------------
# 3. Build a host-native protoc to generate the ONNX protobuf bindings
# ---------------------------------------------------------------------
HOST_PROTOC="${NATIVE_BUILD_DIR}/protobuf/protoc"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    HOST_PROTOC="${NATIVE_BUILD_DIR}/protobuf/protoc.exe"
fi

if [ ! -f "${HOST_PROTOC}" ]; then
    echo "[build_wasm] Building host-native protoc..."
    if [ -d "${NATIVE_BUILD_DIR}" ]; then
        rm -rf "${NATIVE_BUILD_DIR}"
    fi
    mkdir -p "${NATIVE_BUILD_DIR}"

    cmake -S "${PROJECT_ROOT}" -B "${NATIVE_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_EXAMPLES=OFF \
        -Dprotobuf_WITH_ZLIB=OFF \
        -DBENCHMARK_ENABLE_TESTING=OFF \
        "${ABSL_FETCH_FLAG}"

    cmake --build "${NATIVE_BUILD_DIR}" --target protoc \
        -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
fi

if [ ! -f "${HOST_PROTOC}" ]; then
    echo "[build_wasm] ERROR: host protoc not found at ${HOST_PROTOC}."
    exit 1
fi

# ---------------------------------------------------------------------
# 4. Clean and configure the WASM build
# ---------------------------------------------------------------------
mkdir -p "${BUILD_DIR}"
rm -rf "${BUILD_DIR}"/*

echo "[build_wasm] Configuring with emcmake..."
emcmake cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
    -Dprotobuf_WITH_ZLIB=OFF \
    -DBENCHMARK_ENABLE_TESTING=OFF \
    -DONNX2C_PROTOC_EXECUTABLE="${HOST_PROTOC}" \
    "${ABSL_FETCH_FLAG}"

# ---------------------------------------------------------------------
# 5. Build the WASM target
# ---------------------------------------------------------------------
echo "[build_wasm] Building onnx2c_wasm..."
emmake cmake --build "${BUILD_DIR}" --target onnx2c_wasm \
    -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"

# ---------------------------------------------------------------------
# 6. Copy artifacts next to the static frontend
# ---------------------------------------------------------------------
echo "[build_wasm] Copying artifacts to ${OUTPUT_DIR}..."
cp "${BUILD_DIR}/onnx2c.js" "${OUTPUT_DIR}/onnx2c.js"
cp "${BUILD_DIR}/onnx2c.wasm" "${OUTPUT_DIR}/onnx2c.wasm"

echo ""
echo "[build_wasm] Build complete. Artifacts copied to ${OUTPUT_DIR}"
echo "[build_wasm] Serve the folder with any static file server, e.g.:"
echo "  cd ${OUTPUT_DIR}"
echo "  python3 -m http.server 8000"
