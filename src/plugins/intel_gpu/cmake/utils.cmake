# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    if(GPU_RT_TYPE STREQUAL "L0")
        target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_ZE_RT=1)
        target_link_libraries(${TARGET_NAME} PUBLIC LevelZero::LevelZero)
    elseif(GPU_RT_TYPE STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_OCL_RT=1)
        # Do not link OpenCL as It is already linked to the targets that require it
        # target_link_libraries(${TARGET_NAME} PUBLIC OpenCL::NewHeaders OpenCL::OpenCL)
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RT_TYPE}` Only `L0` and `OCL` are supported")
    endif()
endfunction()
