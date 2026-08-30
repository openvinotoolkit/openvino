# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_GPU_SUPPORTED_RUNTIMES ZE OCL SYCL VULKAN)
if(APPLE OR ANDROID OR AARCH64)
    set(OV_GPU_DEFAULT_RT VULKAN)
else()
    set(OV_GPU_DEFAULT_RT OCL)
endif()

ov_option_enum(GPU_RT_TYPE
               "GPU runtime compiled into the plugin (L0 is accepted as a ZE alias)"
               ${OV_GPU_DEFAULT_RT}
               ALLOWED_VALUES ${OV_GPU_SUPPORTED_RUNTIMES} L0)

if(GPU_RT_TYPE STREQUAL "L0")
    set(GPU_RT_TYPE ZE CACHE STRING "GPU runtime compiled into the plugin" FORCE)
endif()

if((APPLE OR ANDROID) AND GPU_RT_TYPE STREQUAL "OCL")
    message(FATAL_ERROR "GPU OCL runtime is not supported on Apple or Android platforms")
endif()

foreach(_ov_gpu_runtime IN LISTS OV_GPU_SUPPORTED_RUNTIMES)
    set(OV_GPU_RUNTIME_${_ov_gpu_runtime}_ENABLED OFF)
endforeach()
set(OV_GPU_RUNTIME_${GPU_RT_TYPE}_ENABLED ON)

set(GPU_ONEDNN_RUNTIME "")
if(GPU_RT_TYPE MATCHES "^(OCL|ZE|SYCL)$")
    set(GPU_ONEDNN_RUNTIME "${GPU_RT_TYPE}")
endif()

if(ANDROID OR
   MINGW OR
   NOT GPU_ONEDNN_RUNTIME OR
   (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    # oneDNN doesn't support old compilers and Android builds for now, so we'll build GPU plugin without oneDNN
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT OFF)
else()
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT ON)
endif()

ov_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ${ENABLE_ONEDNN_FOR_GPU_DEFAULT} "ENABLE_INTEL_GPU" OFF)
ov_dependent_option (ENABLE_CM_FOR_GPU "Enable C for Metal (CM) kernels at GPU runtime" ON "ENABLE_INTEL_GPU" OFF)
