# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_RUNTIME_VULKAN_ENABLED)
    return()
endif()

find_package(Vulkan 1.3 REQUIRED)
find_host_program(OV_GPU_GLSLC_EXECUTABLE NAMES glslc REQUIRED)
find_host_program(OV_GPU_SPIRV_OPT_EXECUTABLE NAMES spirv-opt REQUIRED)
find_host_program(OV_GPU_SPIRV_VAL_EXECUTABLE NAMES spirv-val REQUIRED)

set(VULKAN_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(VULKAN_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VULKAN_SHADER_GENERATOR "${VULKAN_KERNEL_DIR}/generate_spirv_header.py")
file(GLOB VULKAN_SHADER_SOURCES CONFIGURE_DEPENDS "${VULKAN_KERNEL_DIR}/*.comp")

foreach(shader_source IN LISTS VULKAN_SHADER_SOURCES)
    get_filename_component(shader_name "${shader_source}" NAME)
    get_filename_component(shader_stem "${shader_source}" NAME_WE)
    set(raw_spirv "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.raw.spv")
    set(optimized_spirv "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv")
    set(generated_header "${VULKAN_GENERATED_INCLUDE_DIR}/${shader_stem}_spirv.hpp")

    add_custom_command(
        OUTPUT "${generated_header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${VULKAN_GENERATED_INCLUDE_DIR}"
        COMMAND "${OV_GPU_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -I "${VULKAN_KERNEL_DIR}" "${shader_source}" -o "${raw_spirv}"
        COMMAND "${OV_GPU_SPIRV_OPT_EXECUTABLE}" --eliminate-dead-code-aggressive "${raw_spirv}" -o "${optimized_spirv}"
        COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${optimized_spirv}"
        COMMAND "${Python3_EXECUTABLE}" "${VULKAN_SHADER_GENERATOR}"
                "${optimized_spirv}" "${generated_header}" "${shader_stem}_spirv" "kernels/${shader_name}"
        DEPENDS
            "${shader_source}"
            "${VULKAN_KERNEL_DIR}/eltwise_common.glsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc"
            "${VULKAN_SHADER_GENERATOR}"
        COMMENT "Generating host-compiled Vulkan shader ${shader_name}"
        VERBATIM)
    list(APPEND VULKAN_GENERATED_HEADERS "${generated_header}")
endforeach()

set_property(SOURCE ${VULKAN_GENERATED_HEADERS} PROPERTY GENERATED TRUE)
add_custom_target(openvino_intel_gpu_vulkan_shaders DEPENDS ${VULKAN_GENERATED_HEADERS})

set(TARGET_NAME "openvino_intel_gpu_vulkan_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    INCLUDES "${VULKAN_GENERATED_INCLUDE_DIR}"
    LINK_LIBRARIES Vulkan::Vulkan
)
add_dependencies(${TARGET_NAME} openvino_intel_gpu_vulkan_shaders)

ov_build_target_faster(${TARGET_NAME} PCH)
