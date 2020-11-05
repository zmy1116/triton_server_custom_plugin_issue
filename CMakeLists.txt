# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TestPlugins LANGUAGES CXX CUDA)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(CUDA_VERSION 11.1)
set_ifndef(CUDA_ROOT /usr/local/cuda-${CUDA_VERSION})
set_ifndef(CUDNN_ROOT ${CUDA_ROOT})
set_ifndef(TRT_LIB_DIR /usr/include/x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR /usr/lib/x86_64-linux-gnu)
set_ifndef(CUB_DIR /workspace/ubuntu/cub-1.8.0)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

find_library(_CUDA_LIB cudart HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

find_library(_CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

find_path(_CUDNN_INC_DIR cudnn.h HINTS ${CUDNN_ROOT} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(CUDNN_INC_DIR ${_CUDNN_INC_DIR})

find_library(_CUDNN_LIB cudnn HINTS ${CUDNN_ROOT} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(CUDNN_LIB ${_CUDNN_LIB})

find_library(_TRT_INC_DIR NvInfer.h HINTS ${TRT_INC_DIR} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR ${_TRT_INC_DIR})

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVPARSERS_LIB nvparsers HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVPARSERS_LIB ${_NVPARSERS_LIB})

find_library(_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_PLUGIN_LIB ${_NVINFER_PLUGIN_LIB})


# -------- BUILDING --------

# Add include directories
include_directories(
    ${CUDA_INC_DIR}
    ${TRT_INCLUDE}
    ${TRT_INC_DIR}
    ${CUDNN_LIB}
    ${CMAKE_SOURCE_DIR}/plugin/
    ${CUB_DIR}
)


# Define clip plugin library target
add_library(testplugins MODULE
  ${CMAKE_SOURCE_DIR}/plugin/maskRCNNKernels.h
  ${CMAKE_SOURCE_DIR}/plugin/maskRCNNKernels.cu
  ${CMAKE_SOURCE_DIR}/plugin/detectionLayerPlugin.cpp
  ${CMAKE_SOURCE_DIR}/plugin/detectionLayerPlugin.h
  ${CMAKE_SOURCE_DIR}/plugin/checkMacrosPlugin.h
  ${CMAKE_SOURCE_DIR}/plugin/checkMacrosPlugin.cpp
)

# Use C++11
target_compile_features(testplugins PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(testplugins PRIVATE ${NVINFER_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(testplugins PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
