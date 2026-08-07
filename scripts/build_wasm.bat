@echo off
setlocal EnableDelayedExpansion

:: Build onnx2c as a WebAssembly module for static browser deployment.
:: The script automatically installs the Emscripten SDK if it is not found,
:: and builds a host-native protoc for generating the ONNX protobuf bindings.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "BUILD_DIR=%PROJECT_ROOT%\build-wasm"
set "NATIVE_BUILD_DIR=%PROJECT_ROOT%\build-native-protoc"
set "OUTPUT_DIR=%PROJECT_ROOT%\web_converter\wasm"
set "EMSDK_DIR=%PROJECT_ROOT%\emsdk"

:: absl is fetched by protobuf via FetchContent at configure time. We pin it to
:: a fixed local checkout so that, once downloaded, WASM builds work fully offline.
set "ABSL_SOURCE_DIR=%PROJECT_ROOT%\third_party\abseil-cpp"
set "ABSL_GIT_TAG=20250512.1"

cd /d "%PROJECT_ROOT%"

:: ---------------------------------------------------------------------
:: 1. Locate or install the Emscripten SDK
:: ---------------------------------------------------------------------
where emcc >nul 2>&1
if errorlevel 1 (
    echo [build_wasm] Emscripten SDK not found in PATH.
    echo [build_wasm] Checking local SDK at %EMSDK_DIR%...

    if not exist "%EMSDK_DIR%\emsdk.bat" (
        echo [build_wasm] Cloning Emscripten SDK...
        git clone https://github.com/emscripten-core/emsdk.git "%EMSDK_DIR%"
        if errorlevel 1 (
            echo [build_wasm] ERROR: failed to clone emsdk repository.
            exit /b 1
        )
    )

    cd /d "%EMSDK_DIR%"

    echo [build_wasm] Installing latest Emscripten SDK...
    call emsdk.bat install latest
    if errorlevel 1 (
        echo [build_wasm] ERROR: emsdk install failed.
        exit /b 1
    )

    echo [build_wasm] Activating latest Emscripten SDK...
    call emsdk.bat activate latest
    if errorlevel 1 (
        echo [build_wasm] ERROR: emsdk activate failed.
        exit /b 1
    )

    echo [build_wasm] Setting up Emscripten environment...
    call emsdk_env.bat
    if errorlevel 1 (
        echo [build_wasm] ERROR: emsdk_env.bat failed.
        exit /b 1
    )
)

where emcc >nul 2>&1
if errorlevel 1 (
    echo [build_wasm] ERROR: emcc still not available after SDK setup.
    exit /b 1
)

cd /d "%PROJECT_ROOT%"

:: ---------------------------------------------------------------------
:: 2. Ensure Abseil (absl) is available locally for offline builds
:: ---------------------------------------------------------------------
:: protobuf pulls absl via FetchContent during cmake configure. By pointing
:: FETCHCONTENT_SOURCE_DIR_ABSL at a persistent checkout we download once and
:: can rebuild without a network connection afterwards.
set "ABSL_READY=0"
if exist "%ABSL_SOURCE_DIR%\CMakeLists.txt" set "ABSL_READY=1"

if "%ABSL_READY%"=="0" (
    echo [build_wasm] Abseil not found locally, downloading (tag %ABSL_GIT_TAG%^)...
    if exist "%ABSL_SOURCE_DIR%" rmdir /s /q "%ABSL_SOURCE_DIR%"
    git clone --depth 1 --branch "%ABSL_GIT_TAG%" https://github.com/abseil/abseil-cpp.git "%ABSL_SOURCE_DIR%"
    if errorlevel 1 (
        echo [build_wasm] ERROR: failed to download abseil-cpp.
        echo [build_wasm]        An internet connection is required on first run.
        exit /b 1
    )
) else (
    echo [build_wasm] Using local Abseil at %ABSL_SOURCE_DIR% ^(offline-ready^).
)

set "ABSL_FETCH_FLAG=-DFETCHCONTENT_SOURCE_DIR_ABSL=%ABSL_SOURCE_DIR%"

:: ---------------------------------------------------------------------
:: 3. Build a host-native protoc to generate the ONNX protobuf bindings
:: ---------------------------------------------------------------------
set "HOST_PROTOC=%NATIVE_BUILD_DIR%\protobuf\protoc.exe"

:: Prefer Ninja if available; otherwise let CMake pick the default generator.
set "CMAKE_GENERATOR_FLAG="
where ninja >nul 2>&1
if not errorlevel 1 set "CMAKE_GENERATOR_FLAG=-G Ninja"

if not exist "%HOST_PROTOC%" (
    echo [build_wasm] Building host-native protoc...
    if exist "%NATIVE_BUILD_DIR%" rmdir /s /q "%NATIVE_BUILD_DIR%"
    mkdir "%NATIVE_BUILD_DIR%"

    cmake -S "%PROJECT_ROOT%" -B "%NATIVE_BUILD_DIR%" %CMAKE_GENERATOR_FLAG% ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dprotobuf_BUILD_TESTS=OFF ^
        -Dprotobuf_BUILD_EXAMPLES=OFF ^
        -Dprotobuf_WITH_ZLIB=OFF ^
        -DBENCHMARK_ENABLE_TESTING=OFF ^
        %ABSL_FETCH_FLAG%

    if errorlevel 1 (
        echo [build_wasm] ERROR: native protoc configuration failed.
        exit /b 1
    )

    cmake --build "%NATIVE_BUILD_DIR%" --target protoc

    if errorlevel 1 (
        echo [build_wasm] ERROR: native protoc build failed.
        exit /b 1
    )
)

if not exist "%HOST_PROTOC%" (
    echo [build_wasm] ERROR: host protoc not found at %HOST_PROTOC%.
    exit /b 1
)

:: ---------------------------------------------------------------------
:: 4. Clean and configure the WASM build
:: ---------------------------------------------------------------------
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"

echo [build_wasm] Configuring with emcmake...
emcmake cmake -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" %CMAKE_GENERATOR_FLAG% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -Dprotobuf_BUILD_TESTS=OFF ^
    -Dprotobuf_BUILD_EXAMPLES=OFF ^
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF ^
    -Dprotobuf_WITH_ZLIB=OFF ^
    -DBENCHMARK_ENABLE_TESTING=OFF ^
    -DONNX2C_PROTOC_EXECUTABLE="%HOST_PROTOC%" ^
    %ABSL_FETCH_FLAG%

if errorlevel 1 (
    echo [build_wasm] ERROR: cmake configuration failed.
    exit /b 1
)

:: ---------------------------------------------------------------------
:: 5. Build the WASM target
:: ---------------------------------------------------------------------
echo [build_wasm] Building onnx2c_wasm...
emmake cmake --build "%BUILD_DIR%" --target onnx2c_wasm

if errorlevel 1 (
    echo [build_wasm] ERROR: build failed.
    exit /b 1
)

:: ---------------------------------------------------------------------
:: 6. Copy artifacts next to the static frontend
:: ---------------------------------------------------------------------
echo [build_wasm] Copying artifacts to %OUTPUT_DIR%...
copy /Y "%BUILD_DIR%\onnx2c.js" "%OUTPUT_DIR%\onnx2c.js"
copy /Y "%BUILD_DIR%\onnx2c.wasm" "%OUTPUT_DIR%\onnx2c.wasm"

echo.
echo [build_wasm] Build complete. Artifacts copied to %OUTPUT_DIR%
echo [build_wasm] Serve the folder with any static file server, e.g.:
echo   cd %OUTPUT_DIR%
echo   python -m http.server 8000

endlocal
