# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Applies the runtime-specific compile defs / link libs to a target. The runtime defaults to
# the global GPU_RT_TYPE, but a COMBINED build passes an explicit RUNTIME per twin ("ZE"/"OCL")
# so both stacks can be configured in one pass. COMBINED itself is never a per-target runtime.
function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    set(oneValueArgs RUNTIME)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "" ${ARGN})
    if(ARG_RUNTIME)
        set(runtime "${ARG_RUNTIME}")
    else()
        set(runtime "${GPU_RT_TYPE}")
    endif()

    if(runtime STREQUAL "ZE")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
        target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
    elseif(runtime STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
    elseif(runtime STREQUAL "SYCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_SYCL_RT=1)
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${runtime}` Only `ZE`, `OCL` and `SYCL` are supported")
    endif()
endfunction()
