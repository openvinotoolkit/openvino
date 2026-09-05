# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_gpu_resolve_runtime OUT_VAR)
    set(oneValueArgs RUNTIME)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "" ${ARGN})

    if(ARG_RUNTIME)
        set(runtime "${ARG_RUNTIME}")
    elseif(DEFINED OV_GPU_RT AND NOT OV_GPU_RT STREQUAL "")
        set(runtime "${OV_GPU_RT}")
    else()
        set(runtime "${GPU_RT_TYPE}")
    endif()

    set(${OUT_VAR} "${runtime}" PARENT_SCOPE)
endfunction()

function(ov_gpu_set_runtime_definitions_for TARGET_NAME)
    ov_gpu_resolve_runtime(runtime ${ARGN})

    if(runtime STREQUAL "ZE")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1)
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_ZE_RT=1)
    elseif(runtime STREQUAL "OCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_OCL_RT=1)
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_OCL_RT=1)
    elseif(runtime STREQUAL "SYCL")
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_SYCL_RT=1)
        target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_DEFAULT_SYCL_RT=1)
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${runtime}`")
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
    ov_gpu_resolve_runtime(runtime ${ARGN})

    if(runtime STREQUAL "ZE")
        target_link_libraries(${TARGET_NAME} PRIVATE openvino::zero_loader)
    elseif(runtime STREQUAL "OCL")
        target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
    elseif(runtime STREQUAL "SYCL")
        # SYCL toolchain dependencies are attached by add_sycl_to_target.
    else()
        message(FATAL_ERROR "Invalid GPU runtime type: `${runtime}`")
    endif()
endfunction()

function(ov_gpu_set_runtime_interface_for TARGET_NAME)
    ov_gpu_resolve_runtime(runtime ${ARGN})
    ov_gpu_set_runtime_definitions_for(${TARGET_NAME} RUNTIME ${runtime})
    ov_gpu_link_runtime_dependencies_for(${TARGET_NAME} RUNTIME ${runtime})
    ov_gpu_set_opencl_api_version_for(${TARGET_NAME})
endfunction()
