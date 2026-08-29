# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_set_runtime_definitions_for TARGET_NAME)
    if(GPU_RUNTIME_TYPE STREQUAL "ZE")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_ZE_RT=1)
    elseif(GPU_RUNTIME_TYPE STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_OCL_RT=1)
    elseif(GPU_RUNTIME_TYPE STREQUAL "SYCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_SYCL_RT=1)
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_SYCL_RT=1)
    elseif(GPU_RUNTIME_TYPE STREQUAL "VULKAN")
        # The Vulkan default is private to the runtime registry target.
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RUNTIME_TYPE}`")
    endif()
endfunction()

function(ov_gpu_set_opencl_api_version_for TARGET_NAME)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        CL_TARGET_OPENCL_VERSION=${INTEL_GPU_TARGET_OCL_VERSION})
endfunction()

function(ov_gpu_use_header_target_for TARGET_NAME HEADER_TARGET)
    target_include_directories(${TARGET_NAME} PRIVATE
        $<TARGET_PROPERTY:${HEADER_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        $<TARGET_PROPERTY:${HEADER_TARGET},INTERFACE_COMPILE_DEFINITIONS>)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<TARGET_PROPERTY:${HEADER_TARGET},INTERFACE_COMPILE_OPTIONS>)
endfunction()

function(ov_gpu_link_runtime_dependencies_for TARGET_NAME)
    if(GPU_RUNTIME_TYPE STREQUAL "ZE")
        target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
    elseif(GPU_RUNTIME_TYPE STREQUAL "OCL")
        target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
    elseif(GPU_RUNTIME_TYPE STREQUAL "SYCL")
        # SYCL toolchain dependencies are attached by add_sycl_to_target.
    elseif(GPU_RUNTIME_TYPE STREQUAL "VULKAN")
        target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${GPU_RUNTIME_TYPE}`")
    endif()
endfunction()

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    ov_gpu_set_runtime_definitions_for(${TARGET_NAME})
    ov_gpu_link_runtime_dependencies_for(${TARGET_NAME})
    if(OV_GPU_RUNTIME_OCL_ENABLED OR OV_GPU_RUNTIME_ZE_ENABLED OR OV_GPU_RUNTIME_SYCL_ENABLED)
        ov_gpu_set_opencl_api_version_for(${TARGET_NAME})
    endif()
endfunction()
