# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    foreach(GPU_RUNTIME IN LISTS GPU_RUNTIME_TYPES)
        if(GPU_RUNTIME STREQUAL "ZE")
            target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
            target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
        elseif(GPU_RUNTIME STREQUAL "OCL")
            target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
            target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
        elseif(GPU_RUNTIME STREQUAL "SYCL")
            target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_SYCL_RT=1)
        elseif(GPU_RUNTIME STREQUAL "VULKAN")
            # The Vulkan loader is linked only by Vulkan backend targets once
            # they are introduced. Common targets need only the composition tag.
            target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_VULKAN_RT=1)
        else()
            message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RUNTIME}`")
        endif()
    endforeach()

    if(GPU_DEFAULT_RUNTIME STREQUAL "ZE")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_ZE_RT=1)
    elseif(GPU_DEFAULT_RUNTIME STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_OCL_RT=1)
    elseif(GPU_DEFAULT_RUNTIME STREQUAL "SYCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_SYCL_RT=1)
    elseif(GPU_DEFAULT_RUNTIME STREQUAL "VULKAN")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_VULKAN_RT=1)
    else()
        message(FATAL_ERROR "Invalid default GPU runtime type: `${GPU_DEFAULT_RUNTIME}`")
    endif()
endfunction()
