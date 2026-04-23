# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    get_target_property(target_type ${TARGET_NAME} TYPE)
    set(linkable_target_types STATIC_LIBRARY SHARED_LIBRARY MODULE_LIBRARY EXECUTABLE)

    if(GPU_RT_TYPE STREQUAL "L0")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
        if(target_type IN_LIST linkable_target_types)
            target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
        endif()
    elseif(GPU_RT_TYPE STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
        if(target_type IN_LIST linkable_target_types)
            target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
        endif()
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RT_TYPE}` Only `L0` and `OCL` are supported")
    endif()
endfunction()
