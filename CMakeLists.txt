cmake_minimum_required(VERSION 3.18)
project(cuda_swap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate the CUDA driver headers (cuda.h).
# We do NOT link libcuda.so — all symbols are resolved at runtime via
# dlsym(RTLD_NEXT, ...).  We only need the headers at compile time.
#
# Search order:
#   1. CUDA_INCLUDE_DIR cmake variable  (cmake -DCUDA_INCLUDE_DIR=/path)
#   2. CUDAToolkit if available
#   3. Common installation paths
# ---------------------------------------------------------------------------

if(NOT DEFINED CUDA_INCLUDE_DIR)
    # Try find_package first; don't fatal-error if absent.
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
        message(STATUS "cuda_swap: using CUDAToolkit headers at ${CUDA_INCLUDE_DIR}")
    else()
        foreach(candidate
                /usr/local/cuda/include
                /usr/cuda/include
                /opt/cuda/include
                /usr/include/cuda)
            if(EXISTS "${candidate}/cuda.h")
                set(CUDA_INCLUDE_DIR "${candidate}")
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT CUDA_INCLUDE_DIR OR NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR
        "Cannot find cuda.h. Pass -DCUDA_INCLUDE_DIR=/path/to/cuda/include "
        "or install the CUDA toolkit.")
endif()

message(STATUS "cuda_swap: CUDA headers at ${CUDA_INCLUDE_DIR}")

add_library(cuda_swap SHARED
    src/hooks.cpp
    src/vmm.cpp
)

target_include_directories(cuda_swap PRIVATE
    src
    ${CUDA_INCLUDE_DIR}
)

# Link dl only — do NOT link libcuda.so or libcudart.so.
target_link_libraries(cuda_swap PRIVATE dl)

target_compile_options(cuda_swap PRIVATE
    -fvisibility=default
    -O2
    -Wall
    -Wextra
    -Wno-unused-parameter
)

set_target_properties(cuda_swap PROPERTIES
    OUTPUT_NAME cuda_swap
    PREFIX ""
)
