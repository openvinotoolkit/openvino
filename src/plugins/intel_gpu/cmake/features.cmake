# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# GPU_RUNTIME_TYPES selects the compiled runtimes. GPU_DEFAULT_RUNTIME selects
# the default one. GPU_RT_TYPE remains as a backwards-compatible alias for the
# default runtime.
set(OV_GPU_SUPPORTED_RUNTIMES ZE OCL SYCL VULKAN)
if(APPLE OR ANDROID)
    set(OV_GPU_DEFAULT_RT "VULKAN")
else()
    set(OV_GPU_DEFAULT_RT "OCL")
endif()

ov_option_enum(GPU_RT_TYPE
               "Legacy default GPU runtime. Supported values: OCL, SYCL, ZE, VULKAN (L0 is accepted as ZE alias)"
               ${OV_GPU_DEFAULT_RT}
               ALLOWED_VALUES ZE OCL L0 SYCL VULKAN)

if(GPU_RT_TYPE STREQUAL "L0")
    set(GPU_RT_TYPE "ZE" CACHE STRING "Legacy default GPU runtime" FORCE)
endif()

set(GPU_RUNTIME_TYPES "${GPU_RT_TYPE}" CACHE STRING
    "Semicolon-separated GPU runtimes compiled into the GPU plugin")
set(GPU_DEFAULT_RUNTIME "${GPU_RT_TYPE}" CACHE STRING
    "Default GPU runtime; must be present in GPU_RUNTIME_TYPES")
set_property(CACHE GPU_DEFAULT_RUNTIME PROPERTY STRINGS ${OV_GPU_SUPPORTED_RUNTIMES})
list(APPEND OV_OPTIONS GPU_RUNTIME_TYPES GPU_DEFAULT_RUNTIME)
set(OV_OPTIONS "${OV_OPTIONS}" CACHE INTERNAL "A list of OpenVINO cmake options")

# Normalize the compiled runtime list before validating it.
set(_ov_gpu_normalized_runtimes ${GPU_RUNTIME_TYPES})
list(TRANSFORM _ov_gpu_normalized_runtimes STRIP)
list(TRANSFORM _ov_gpu_normalized_runtimes TOUPPER)
list(TRANSFORM _ov_gpu_normalized_runtimes REPLACE "^L0$" "ZE")
list(REMOVE_DUPLICATES _ov_gpu_normalized_runtimes)

if(NOT _ov_gpu_normalized_runtimes)
    message(FATAL_ERROR "GPU_RUNTIME_TYPES must contain at least one runtime")
endif()

foreach(_ov_gpu_runtime IN LISTS _ov_gpu_normalized_runtimes)
    if(NOT _ov_gpu_runtime IN_LIST OV_GPU_SUPPORTED_RUNTIMES)
        message(FATAL_ERROR
            "Unsupported GPU runtime '${_ov_gpu_runtime}'. Supported values: ${OV_GPU_SUPPORTED_RUNTIMES}")
    endif()
endforeach()

if((APPLE OR ANDROID) AND "OCL" IN_LIST _ov_gpu_normalized_runtimes)
    message(FATAL_ERROR "GPU OCL runtime is not supported on Apple or Android platforms")
endif()

set(GPU_RUNTIME_TYPES "${_ov_gpu_normalized_runtimes}" CACHE STRING
    "Semicolon-separated GPU runtimes compiled into the GPU plugin" FORCE)

# Normalize and validate the default runtime.
string(STRIP "${GPU_DEFAULT_RUNTIME}" GPU_DEFAULT_RUNTIME)
string(TOUPPER "${GPU_DEFAULT_RUNTIME}" GPU_DEFAULT_RUNTIME)
if(GPU_DEFAULT_RUNTIME STREQUAL "L0")
    set(GPU_DEFAULT_RUNTIME "ZE")
endif()
if(NOT GPU_DEFAULT_RUNTIME IN_LIST OV_GPU_SUPPORTED_RUNTIMES)
    message(FATAL_ERROR
        "Unsupported GPU_DEFAULT_RUNTIME '${GPU_DEFAULT_RUNTIME}'. Supported values: ${OV_GPU_SUPPORTED_RUNTIMES}")
endif()
if(NOT GPU_DEFAULT_RUNTIME IN_LIST GPU_RUNTIME_TYPES)
    message(FATAL_ERROR
        "GPU_DEFAULT_RUNTIME '${GPU_DEFAULT_RUNTIME}' must be present in GPU_RUNTIME_TYPES '${GPU_RUNTIME_TYPES}'")
endif()
set(GPU_DEFAULT_RUNTIME "${GPU_DEFAULT_RUNTIME}" CACHE STRING
    "Default GPU runtime; must be present in GPU_RUNTIME_TYPES" FORCE)
set(GPU_RT_TYPE "${GPU_DEFAULT_RUNTIME}" CACHE STRING "Legacy default GPU runtime" FORCE)

# Expose one boolean for each runtime to keep downstream conditions simple.
foreach(_ov_gpu_runtime IN LISTS OV_GPU_SUPPORTED_RUNTIMES)
    set(OV_GPU_RUNTIME_${_ov_gpu_runtime}_ENABLED OFF)
endforeach()
foreach(_ov_gpu_runtime IN LISTS GPU_RUNTIME_TYPES)
    set(OV_GPU_RUNTIME_${_ov_gpu_runtime}_ENABLED ON)
endforeach()

set(GPU_ONEDNN_RUNTIME "")
if(GPU_DEFAULT_RUNTIME MATCHES "^(OCL|ZE|SYCL)$")
    set(GPU_ONEDNN_RUNTIME "${GPU_DEFAULT_RUNTIME}")
elseif(OV_GPU_RUNTIME_OCL_ENABLED)
    set(GPU_ONEDNN_RUNTIME "OCL")
elseif(OV_GPU_RUNTIME_ZE_ENABLED)
    set(GPU_ONEDNN_RUNTIME "ZE")
elseif(OV_GPU_RUNTIME_SYCL_ENABLED)
    set(GPU_ONEDNN_RUNTIME "SYCL")
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
