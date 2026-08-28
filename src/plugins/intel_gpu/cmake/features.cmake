# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# GPU_RUNTIME_TYPE selects the existing runtime compiled into the GPU plugin.
set(OV_GPU_SUPPORTED_RUNTIMES ZE OCL SYCL)
set(OV_GPU_DEFAULT_RT "OCL")

# Import the established spelling when reconfiguring an existing build tree.
# New configurations expose only the descriptive GPU_RUNTIME_TYPE option.
if(DEFINED GPU_RUNTIME_TYPE)
    set(_ov_gpu_requested_runtime "${GPU_RUNTIME_TYPE}")
elseif(DEFINED GPU_RT_TYPE)
    set(_ov_gpu_requested_runtime "${GPU_RT_TYPE}")
else()
    set(_ov_gpu_requested_runtime "${OV_GPU_DEFAULT_RT}")
endif()

list(LENGTH _ov_gpu_requested_runtime _ov_gpu_requested_runtime_count)
if(NOT _ov_gpu_requested_runtime_count EQUAL 1)
    message(FATAL_ERROR "GPU_RUNTIME_TYPE selects exactly one runtime")
endif()

string(STRIP "${_ov_gpu_requested_runtime}" _ov_gpu_requested_runtime)
string(TOUPPER "${_ov_gpu_requested_runtime}" _ov_gpu_requested_runtime)
if(_ov_gpu_requested_runtime STREQUAL "L0")
    set(_ov_gpu_requested_runtime "ZE")
endif()

list(REMOVE_ITEM OV_OPTIONS GPU_RT_TYPE)
set(OV_OPTIONS "${OV_OPTIONS}" CACHE INTERNAL "A list of OpenVINO cmake options" FORCE)

ov_option_enum(GPU_RUNTIME_TYPE
               "GPU runtime compiled into the plugin. Supported values: OCL, SYCL, ZE"
               ${_ov_gpu_requested_runtime}
               ALLOWED_VALUES ZE OCL SYCL)

# Preserve the established spelling for existing automation without exposing
# two runtime-selection options in newly configured build trees.
set(GPU_RT_TYPE "${GPU_RUNTIME_TYPE}" CACHE STRING "Deprecated alias for GPU_RUNTIME_TYPE" FORCE)
set_property(CACHE GPU_RT_TYPE PROPERTY STRINGS ZE OCL SYCL)
mark_as_advanced(GPU_RT_TYPE)

foreach(_ov_gpu_runtime IN LISTS OV_GPU_SUPPORTED_RUNTIMES)
    set(OV_GPU_RUNTIME_${_ov_gpu_runtime}_ENABLED OFF)
endforeach()
set(OV_GPU_RUNTIME_${GPU_RUNTIME_TYPE}_ENABLED ON)

set(GPU_ONEDNN_RUNTIME "${GPU_RUNTIME_TYPE}")
if(ANDROID OR MINGW OR (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    # oneDNN doesn't support old compilers and Android builds for now, so we'll build GPU plugin without oneDNN
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT OFF)
else()
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT ON)
endif()

ov_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ${ENABLE_ONEDNN_FOR_GPU_DEFAULT} "ENABLE_INTEL_GPU" OFF)
ov_dependent_option (ENABLE_CM_FOR_GPU "Enable C for Metal (CM) kernels at GPU runtime" ON "ENABLE_INTEL_GPU" OFF)
