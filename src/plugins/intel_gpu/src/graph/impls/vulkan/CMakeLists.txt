# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_RUNTIME_VULKAN_ENABLED)
    return()
endif()

find_host_program(OV_GPU_CLSPV_EXECUTABLE NAMES clspv REQUIRED)
find_host_program(OV_GPU_CLSPV_REFLECTION_EXECUTABLE NAMES clspv-reflection REQUIRED)
find_host_program(OV_GPU_SLANGC_EXECUTABLE NAMES slangc REQUIRED)
find_host_program(OV_GPU_SPIRV_OPT_EXECUTABLE NAMES spirv-opt REQUIRED)
find_host_program(OV_GPU_SPIRV_VAL_EXECUTABLE NAMES spirv-val REQUIRED)

set(VULKAN_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(VULKAN_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VULKAN_SHADER_GENERATOR "${VULKAN_KERNEL_DIR}/generate_spirv_header.py")
set(VULKAN_CLSPV_REFLECTION_VALIDATOR "${VULKAN_KERNEL_DIR}/validate_clspv_reflection.py")
set(VULKAN_PORTABLE_PUSH_CONSTANT_LIMIT 128)  # Vulkan guarantees at least 128 bytes.
set_property(GLOBAL PROPERTY OV_GPU_VULKAN_GENERATED_HEADERS "")
set_property(GLOBAL PROPERTY OV_GPU_VULKAN_GENERATED_INCLUDE_DIR "${VULKAN_GENERATED_INCLUDE_DIR}")

function(add_vulkan_clspv_kernel)
    cmake_parse_arguments(PARSE_ARGV 0 KERNEL "" "NAME;SOURCE;ENTRY_POINT;DESCRIPTION" "DEFINITIONS;DEPENDENCIES")

    if(NOT KERNEL_NAME OR NOT KERNEL_SOURCE OR NOT KERNEL_ENTRY_POINT)
        message(FATAL_ERROR "A CLSPV Vulkan kernel requires NAME, SOURCE, and ENTRY_POINT")
    endif()
    if(NOT KERNEL_DESCRIPTION)
        set(KERNEL_DESCRIPTION "${KERNEL_SOURCE}")
    endif()

    set(compiler_definitions)
    foreach(definition IN LISTS KERNEL_DEFINITIONS)
        list(APPEND compiler_definitions "-D${definition}")
    endforeach()

    set(raw_spirv "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.reflect.spv")
    set(reflection "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.reflection.csv")
    set(reflection_stamp "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.reflection.validated")
    set(optimized_spirv "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.spv")
    set(generated_header "${VULKAN_GENERATED_INCLUDE_DIR}/${KERNEL_NAME}_spirv.hpp")

    add_custom_command(
        OUTPUT "${generated_header}" "${reflection}" "${reflection_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${VULKAN_GENERATED_INCLUDE_DIR}"
        COMMAND "${OV_GPU_CLSPV_EXECUTABLE}" --spv-version=1.6 -inline-entry-points
                -pod-pushconstant "-max-pushconstant-size=${VULKAN_PORTABLE_PUSH_CONSTANT_LIMIT}"
                ${compiler_definitions} "${KERNEL_SOURCE}" -o "${raw_spirv}"
        COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${raw_spirv}"
        COMMAND "${OV_GPU_CLSPV_REFLECTION_EXECUTABLE}" -d "${raw_spirv}" -o "${reflection}"
        COMMAND "${Python3_EXECUTABLE}" "${VULKAN_CLSPV_REFLECTION_VALIDATOR}"
                "${reflection}" "${reflection_stamp}" "${KERNEL_ENTRY_POINT}"
        COMMAND "${OV_GPU_SPIRV_OPT_EXECUTABLE}" --strip-reflect --eliminate-dead-code-aggressive
                "${raw_spirv}" -o "${optimized_spirv}"
        COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${optimized_spirv}"
        COMMAND "${Python3_EXECUTABLE}" "${VULKAN_SHADER_GENERATOR}"
                "${optimized_spirv}" "${generated_header}" "${KERNEL_NAME}_spirv" "${KERNEL_DESCRIPTION}"
        DEPENDS
            "${KERNEL_SOURCE}"
            ${KERNEL_DEPENDENCIES}
            "${OV_GPU_CLSPV_EXECUTABLE}"
            "${OV_GPU_CLSPV_REFLECTION_EXECUTABLE}"
            "${OV_GPU_SPIRV_OPT_EXECUTABLE}"
            "${OV_GPU_SPIRV_VAL_EXECUTABLE}"
            "${VULKAN_CLSPV_REFLECTION_VALIDATOR}"
            "${VULKAN_SHADER_GENERATOR}"
        COMMENT "Generating host-compiled CLSPV Vulkan kernel ${KERNEL_NAME}"
        VERBATIM)

    set_property(GLOBAL APPEND PROPERTY OV_GPU_VULKAN_GENERATED_HEADERS "${generated_header}")
endfunction()

function(add_vulkan_slang_kernel)
    cmake_parse_arguments(PARSE_ARGV 0 KERNEL "" "NAME;SOURCE;ENTRY_POINT;DESCRIPTION" "")

    if(NOT KERNEL_NAME OR NOT KERNEL_SOURCE OR NOT KERNEL_ENTRY_POINT)
        message(FATAL_ERROR "A Slang Vulkan kernel requires NAME, SOURCE, and ENTRY_POINT")
    endif()
    if(NOT KERNEL_DESCRIPTION)
        set(KERNEL_DESCRIPTION "${KERNEL_SOURCE}")
    endif()

    set(raw_spirv "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.raw.spv")
    set(optimized_spirv "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.spv")
    set(generated_header "${VULKAN_GENERATED_INCLUDE_DIR}/${KERNEL_NAME}_spirv.hpp")

    add_custom_command(
        OUTPUT "${generated_header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${VULKAN_GENERATED_INCLUDE_DIR}"
        COMMAND "${OV_GPU_SLANGC_EXECUTABLE}" "${KERNEL_SOURCE}"
                -target spirv -profile spirv_1_6 -stage compute -entry "${KERNEL_ENTRY_POINT}"
                -emit-spirv-directly -fvk-use-entrypoint-name -O3 -o "${raw_spirv}"
        COMMAND "${OV_GPU_SPIRV_OPT_EXECUTABLE}" --eliminate-dead-code-aggressive
                "${raw_spirv}" -o "${optimized_spirv}"
        COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${optimized_spirv}"
        COMMAND "${Python3_EXECUTABLE}" "${VULKAN_SHADER_GENERATOR}"
                "${optimized_spirv}" "${generated_header}" "${KERNEL_NAME}_spirv" "${KERNEL_DESCRIPTION}"
        DEPENDS
            "${KERNEL_SOURCE}"
            "${OV_GPU_SLANGC_EXECUTABLE}"
            "${OV_GPU_SPIRV_OPT_EXECUTABLE}"
            "${OV_GPU_SPIRV_VAL_EXECUTABLE}"
            "${VULKAN_SHADER_GENERATOR}"
        COMMENT "Generating host-compiled Slang Vulkan kernel ${KERNEL_NAME}"
        VERBATIM)

    set_property(GLOBAL APPEND PROPERTY OV_GPU_VULKAN_GENERATED_HEADERS "${generated_header}")
endfunction()

# Build-time compiler contracts only. Production Eltwise is materialized from
# kernel_selector and compiled in-process while the model is compiled.
add_vulkan_clspv_kernel(
    NAME clspv_bootstrap
    SOURCE "${VULKAN_KERNEL_DIR}/clspv_bootstrap.cl"
    ENTRY_POINT clspv_bootstrap
    DESCRIPTION "kernels/clspv_bootstrap.cl"
)
add_vulkan_slang_kernel(
    NAME slang_bootstrap
    SOURCE "${VULKAN_KERNEL_DIR}/slang_bootstrap.slang"
    ENTRY_POINT slang_bootstrap
    DESCRIPTION "kernels/slang_bootstrap.slang"
)

get_property(VULKAN_GENERATED_HEADERS GLOBAL PROPERTY OV_GPU_VULKAN_GENERATED_HEADERS)
set_property(SOURCE ${VULKAN_GENERATED_HEADERS} PROPERTY GENERATED TRUE)
add_custom_target(openvino_intel_gpu_vulkan_shaders${OV_GPU_SUFFIX} DEPENDS ${VULKAN_GENERATED_HEADERS})

set(TARGET_NAME "openvino_intel_gpu_vulkan_obj${OV_GPU_SUFFIX}")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    INCLUDES "${VULKAN_GENERATED_INCLUDE_DIR}"
    LINK_LIBRARIES Vulkan::Vulkan
)
add_dependencies(${TARGET_NAME} openvino_intel_gpu_vulkan_shaders${OV_GPU_SUFFIX})

ov_build_target_faster(${TARGET_NAME} PCH)
