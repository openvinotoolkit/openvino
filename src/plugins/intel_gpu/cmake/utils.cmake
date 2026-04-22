# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    get_target_property(_ov_gpu_target_type ${TARGET_NAME} TYPE)
    set(_ov_gpu_skip_link FALSE)
    if(_ov_gpu_target_type STREQUAL "OBJECT_LIBRARY")
        set(_ov_gpu_skip_link TRUE)
    endif()

    if(GPU_RT_TYPE STREQUAL "L0")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
        if(NOT _ov_gpu_skip_link)
            target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
        else()
            target_include_directories(${TARGET_NAME} PRIVATE
                $<TARGET_PROPERTY:openvino::zero_loader,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:LevelZero::Headers,INTERFACE_INCLUDE_DIRECTORIES>)
        endif()
    elseif(GPU_RT_TYPE STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
        if(NOT _ov_gpu_skip_link)
            target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
        else()
            target_include_directories(${TARGET_NAME} PRIVATE
                $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>)
        endif()
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RT_TYPE}` Only `L0` and `OCL` are supported")
    endif()
endfunction()
