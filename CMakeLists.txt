cmake_minimum_required(VERSION 3.18)
project(bitnet_nvfp4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include)
  message(FATAL_ERROR "CUTLASS submodule missing. Run: git submodule update --init --recursive")
endif()

# NVFP4 block-scaled GEMM requires SM120a+. Force a sane default unless user overrides.
if (NOT CMAKE_CUDA_ARCHITECTURES OR
    NOT CMAKE_CUDA_ARCHITECTURES MATCHES "120a|120|121")
  message(STATUS "Forcing CUDA_ARCHITECTURES=120a for NVFP4 build")
  set(CMAKE_CUDA_ARCHITECTURES "120a" CACHE STRING "" FORCE)
endif()

add_library(bitnet_nvfp4_cuda
    src/cuda_kernels.cu
    src/efla_lm_kernels.cu
)

target_include_directories(bitnet_nvfp4_cuda PUBLIC src)
target_link_libraries(bitnet_nvfp4_cuda PUBLIC CUDA::cudart)
if (TARGET CUDA::cuda_driver)
  target_link_libraries(bitnet_nvfp4_cuda PUBLIC CUDA::cuda_driver)
elseif (TARGET CUDA::cuda)
  target_link_libraries(bitnet_nvfp4_cuda PUBLIC CUDA::cuda)
endif()

target_compile_options(bitnet_nvfp4_cuda PRIVATE -O3)
set_target_properties(bitnet_nvfp4_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include)
  target_include_directories(bitnet_nvfp4_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include)
    target_include_directories(bitnet_nvfp4_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include)
  endif()
  target_compile_definitions(bitnet_nvfp4_cuda PUBLIC BITNET_EGGROLL_HAS_CUTLASS)
  target_compile_options(bitnet_nvfp4_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  # CUDA_ARCHITECTURES already forced above for NVFP4 builds.
endif()

add_executable(train_efla_lm
    src/train_efla_lm.cpp
    src/dataset.cpp
    src/rng.cpp
)

target_include_directories(train_efla_lm PRIVATE src)
target_link_libraries(train_efla_lm PRIVATE bitnet_nvfp4_cuda)
if(OpenMP_CXX_FOUND)
  target_link_libraries(train_efla_lm PRIVATE OpenMP::OpenMP_CXX)
endif()

find_path(NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if (NVTX3_INCLUDE_DIR)
  target_include_directories(train_efla_lm PRIVATE ${NVTX3_INCLUDE_DIR})
  target_compile_definitions(train_efla_lm PRIVATE BITNET_EGGROLL_HAS_NVTX)
else()
  find_path(NVTX_INCLUDE_DIR nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
  if (NVTX_INCLUDE_DIR)
    target_include_directories(train_efla_lm PRIVATE ${NVTX_INCLUDE_DIR})
    target_compile_definitions(train_efla_lm PRIVATE BITNET_EGGROLL_HAS_NVTX)
  endif()
  if (TARGET CUDA::nvToolsExt)
    target_link_libraries(train_efla_lm PRIVATE CUDA::nvToolsExt)
  else()
    find_library(NVTOOLSEXT_LIBRARY nvToolsExt)
    if (NVTOOLSEXT_LIBRARY)
      target_link_libraries(train_efla_lm PRIVATE ${NVTOOLSEXT_LIBRARY})
    endif()
  endif()
endif()

target_compile_options(train_efla_lm PRIVATE -O3)
