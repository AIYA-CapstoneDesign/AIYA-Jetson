cmake_minimum_required(VERSION 3.18)
project(AIYA-Jetson LANGUAGES CXX CUDA)

# ----------------------
# Build Configuration
# ----------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)

# Set CUDA architectures for Jetson
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin series
endif()

# ----------------------
# Find Required Packages
# ----------------------
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED 
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-webrtc-1.0
    gstreamer-sdp-1.0
    gstreamer-pbutils-1.0
)

# libsoup and json-glib for HTTP and JSON processing
# Try libsoup-3.0 first, fallback to libsoup-2.4
pkg_check_modules(LIBSOUP libsoup-3.0)
if(NOT LIBSOUP_FOUND)
    pkg_check_modules(LIBSOUP REQUIRED libsoup-2.4)
    message(STATUS "Using libsoup-2.4 (libsoup-3.0 not found)")
else()
    message(STATUS "Using libsoup-3.0")
endif()

pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)

# TensorRT - Improved detection
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS 
        ENV TENSORRT_ROOT
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
        /opt/tensorrt/include
    PATH_SUFFIXES include
)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS 
        ENV TENSORRT_ROOT
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /opt/tensorrt/lib
    PATH_SUFFIXES lib lib64
)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS 
        ENV TENSORRT_ROOT
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /opt/tensorrt/lib
    PATH_SUFFIXES lib lib64
)

# Verify TensorRT was found
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER OR NOT TENSORRT_LIBRARY_INFER_PLUGIN)
    message(FATAL_ERROR "TensorRT not found. Please install TensorRT or set TENSORRT_ROOT environment variable.")
endif()

# ----------------------
# Collect Source Files
# ----------------------
file(GLOB_RECURSE CPP_SOURCES 
    "src/*.cpp"
    "cuda/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
    "cuda/*.cu"
    "src/*.cu"
)

# Combine C++ and CUDA sources
set(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

# Add CUDA kernels (if depth_colormap.cu exists, otherwise remove this line)
# set(CUDA_SOURCES "depth_colormap.cu")

# ----------------------
# Create Executable
# ----------------------
add_executable(aiya-jetson ${SOURCES})

# Set properties
set_target_properties(aiya-jetson PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ----------------------
# Include Directories
# ----------------------
# Function to recursively collect all subdirectories
function(get_all_subdirectories result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${curdir}/${child})
            list(APPEND dirlist ${curdir}/${child})
            get_all_subdirectories(subdirs ${curdir}/${child})
            list(APPEND dirlist ${subdirs})
        endif()
    endforeach()
    set(${result} ${dirlist} PARENT_SCOPE)
endfunction()

# Base directories
set(BASE_INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

# Collect all subdirectories recursively
set(ALL_INCLUDE_DIRS ${BASE_INCLUDE_DIRS})
foreach(BASE_DIR ${BASE_INCLUDE_DIRS})
    if(EXISTS ${BASE_DIR})
        get_all_subdirectories(SUBDIRS ${BASE_DIR})
        list(APPEND ALL_INCLUDE_DIRS ${SUBDIRS})
    endif()
endforeach()

# Remove duplicates and set include directories
list(REMOVE_DUPLICATES ALL_INCLUDE_DIRS)

target_include_directories(aiya-jetson
    PRIVATE
        ${ALL_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${GST_INCLUDE_DIRS}
        ${LIBSOUP_INCLUDE_DIRS}
        ${JSON_GLIB_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${X11_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS}
)

# ----------------------
# Compile Definitions
# ----------------------
# Remove problematic compile definitions that contain compiler flags
# target_compile_definitions will be handled through target_compile_options instead

# ----------------------
# Link Libraries
# ----------------------
target_link_libraries(aiya-jetson
    PRIVATE
        yaml-cpp
        ${OpenCV_LIBS}
        ${GST_LIBRARIES}
        ${LIBSOUP_LIBRARIES}
        ${JSON_GLIB_LIBRARIES}
        ${TENSORRT_LIBRARY_INFER}
        ${TENSORRT_LIBRARY_INFER_PLUGIN}
        CUDA::cudart
        CUDA::curand
        CUDA::nppc
        CUDA::nppig
        CUDA::nppicc
        CUDA::nppidei
        CUDA::nppif
        CUDA::nppim
        CUDA::nppist
        CUDA::nppisu
        CUDA::nppitc
        OpenGL::GL
        OpenGL::GLU
        ${GLU_LIBRARY}
        GLEW::GLEW
        ${X11_LIBRARIES}
        X11
)

# ----------------------
# Compiler Options
# ----------------------
target_compile_options(aiya-jetson
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall 
            -Wextra 
            -Wpedantic
            -Wno-overflow
            -Wno-sign-compare
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-comment
            -Wno-variadic-macros
            -Wno-switch
            -pthread
            -O3
            -DNDEBUG
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xcompiler -fPIC
        >
)

# ----------------------
# Link Options
# ----------------------
target_link_options(aiya-jetson
    PRIVATE
        ${GST_LDFLAGS}
        ${LIBSOUP_LDFLAGS}
        ${JSON_GLIB_LDFLAGS}
)

# ----------------------
# Install Configuration
# ----------------------
install(TARGETS aiya-jetson
    RUNTIME DESTINATION bin
)

# Optional: Print configuration summary
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libraries: ${TENSORRT_LIBRARY_INFER}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "libsoup version: ${LIBSOUP_VERSION}")
message(STATUS "json-glib version: ${JSON_GLIB_VERSION}")
message(STATUS "====================================")

# OpenGL, X11, GLU, GLEW
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 QUIET)

# Check for GLU
find_library(GLU_LIBRARY GLU REQUIRED)
if(NOT GLU_LIBRARY)
    message(FATAL_ERROR "GLU library not found")
endif()