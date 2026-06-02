# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    if(GPU_RT_TYPE STREQUAL "ZE")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
        target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
    elseif(GPU_RT_TYPE STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
    elseif(GPU_RT_TYPE STREQUAL "SYCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_SYCL_RT=1)
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RT_TYPE}` Only `ZE`, `OCL` and `SYCL` are supported")
    endif()
endfunction()
