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

    if(shader_name MATCHES "^eltwise_(dense|fused_dense).*\\.comp$")
        set(restrict_shader_stem "${shader_stem}_restrict")
        set(restrict_raw_spirv "${CMAKE_CURRENT_BINARY_DIR}/${restrict_shader_stem}.raw.spv")
        set(restrict_optimized_spirv "${CMAKE_CURRENT_BINARY_DIR}/${restrict_shader_stem}.spv")
        set(restrict_generated_header "${VULKAN_GENERATED_INCLUDE_DIR}/${restrict_shader_stem}_spirv.hpp")

        add_custom_command(
            OUTPUT "${restrict_generated_header}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${VULKAN_GENERATED_INCLUDE_DIR}"
            COMMAND "${OV_GPU_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -I "${VULKAN_KERNEL_DIR}"
                    -DELTWISE_RESTRICT_OUTPUT=1 "${shader_source}" -o "${restrict_raw_spirv}"
            COMMAND "${OV_GPU_SPIRV_OPT_EXECUTABLE}" --eliminate-dead-code-aggressive
                    "${restrict_raw_spirv}" -o "${restrict_optimized_spirv}"
            COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${restrict_optimized_spirv}"
            COMMAND "${Python3_EXECUTABLE}" "${VULKAN_SHADER_GENERATOR}"
                    "${restrict_optimized_spirv}" "${restrict_generated_header}" "${restrict_shader_stem}_spirv"
                    "kernels/${shader_name} with runtime-proven non-aliasing output"
            DEPENDS
                "${shader_source}"
                "${VULKAN_KERNEL_DIR}/eltwise_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc"
                "${VULKAN_SHADER_GENERATOR}"
            COMMENT "Generating host-compiled Vulkan shader ${restrict_shader_stem}"
            VERBATIM)
        list(APPEND VULKAN_GENERATED_HEADERS "${restrict_generated_header}")
    endif()
endforeach()

function(add_vulkan_hot_eltwise_shader shader_stem shader_name fixed_mode input0_type input1_type output_type)
    set(restrict_variants FALSE)
    if(shader_name MATCHES "^eltwise_dense.*\\.comp$")
        list(APPEND restrict_variants TRUE)
    endif()
    foreach(restrict_output IN LISTS restrict_variants)
        set(variant_stem "${shader_stem}")
        set(restrict_definition "")
        set(variant_description "")
        if(restrict_output)
            set(variant_stem "${shader_stem}_restrict")
            set(restrict_definition -DELTWISE_RESTRICT_OUTPUT=1)
            set(variant_description " with runtime-proven non-aliasing output")
        endif()
        set(shader_source "${VULKAN_KERNEL_DIR}/${shader_name}")
        set(raw_spirv "${CMAKE_CURRENT_BINARY_DIR}/${variant_stem}.raw.spv")
        set(optimized_spirv "${CMAKE_CURRENT_BINARY_DIR}/${variant_stem}.spv")
        set(generated_header "${VULKAN_GENERATED_INCLUDE_DIR}/${variant_stem}_spirv.hpp")

        add_custom_command(
            OUTPUT "${generated_header}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${VULKAN_GENERATED_INCLUDE_DIR}"
            COMMAND "${OV_GPU_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -I "${VULKAN_KERNEL_DIR}"
                    "-DELTWISE_FIXED_MODE=${fixed_mode}"
                    "-DELTWISE_FIXED_INPUT0_TYPE=${input0_type}"
                    "-DELTWISE_FIXED_INPUT1_TYPE=${input1_type}"
                    "-DELTWISE_FIXED_OUTPUT_TYPE=${output_type}"
                    ${restrict_definition} "${shader_source}" -o "${raw_spirv}"
            COMMAND "${OV_GPU_SPIRV_OPT_EXECUTABLE}" -O "${raw_spirv}" -o "${optimized_spirv}"
            COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${optimized_spirv}"
            COMMAND "${Python3_EXECUTABLE}" "${VULKAN_SHADER_GENERATOR}"
                    "${optimized_spirv}" "${generated_header}" "${variant_stem}_spirv"
                    "kernels/${shader_name} fixed mode=${fixed_mode} input0=${input0_type} input1=${input1_type} output=${output_type}${variant_description}"
            DEPENDS
                "${shader_source}"
                "${VULKAN_KERNEL_DIR}/eltwise_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc"
                "${VULKAN_SHADER_GENERATOR}"
            COMMENT "Generating host-compiled Vulkan shader ${variant_stem}"
            VERBATIM)
        list(APPEND VULKAN_GENERATED_HEADERS "${generated_header}")
    endforeach()
    set(VULKAN_GENERATED_HEADERS "${VULKAN_GENERATED_HEADERS}" PARENT_SCOPE)
endfunction()

# Bounded common hot set: cheap/expensive arithmetic for f32/i64 dense layouts and
# one comparison for generic/fast f32 broadcast. The complete-family modules remain fallback.
add_vulkan_hot_eltwise_shader(eltwise_dense_f32_sum_push_constants eltwise_dense_push_constants.comp mode_sum type_f32 type_f32 type_f32)
add_vulkan_hot_eltwise_shader(eltwise_dense_f32_div_push_constants eltwise_dense_push_constants.comp mode_div type_f32 type_f32 type_f32)
add_vulkan_hot_eltwise_shader(eltwise_dense_i64_sum_push_constants eltwise_dense_push_constants.comp mode_sum type_i64 type_i64 type_i64)
add_vulkan_hot_eltwise_shader(eltwise_dense_i64_div_push_constants eltwise_dense_push_constants.comp mode_div type_i64 type_i64 type_i64)
add_vulkan_hot_eltwise_shader(eltwise_broadcast_f32_eq eltwise.comp mode_eq type_f32 type_f32 type_boolean)
add_vulkan_hot_eltwise_shader(eltwise_broadcast_fast_f32_eq eltwise_broadcast_fast.comp mode_eq type_f32 type_f32 type_boolean)

function(add_vulkan_fused_chain_shader chain_length vector_width no_tail)
    set(shader_source "${VULKAN_KERNEL_DIR}/eltwise_fused_dense_chain.comp")
    set(shader_stem "eltwise_fused_dense_chain_length${chain_length}")
    set(vector_definitions "")
    if(vector_width GREATER 0)
        string(APPEND shader_stem "_f32_vec${vector_width}")
        list(APPEND vector_definitions "-DELTWISE_F32_VECTOR_WIDTH=${vector_width}")
    endif()
    if(no_tail)
        string(APPEND shader_stem "_no_tail")
        list(APPEND vector_definitions "-DELTWISE_F32_NO_TAIL=1")
    endif()

    foreach(restrict_output FALSE TRUE)
        set(variant_stem "${shader_stem}")
        set(restrict_definition "")
        set(variant_description "")
        if(restrict_output)
            string(APPEND variant_stem "_restrict")
            set(restrict_definition -DELTWISE_RESTRICT_OUTPUT=1)
            set(variant_description " with runtime-proven non-aliasing output")
        endif()
        set(raw_spirv "${CMAKE_CURRENT_BINARY_DIR}/${variant_stem}.raw.spv")
        set(optimized_spirv "${CMAKE_CURRENT_BINARY_DIR}/${variant_stem}.spv")
        set(generated_header "${VULKAN_GENERATED_INCLUDE_DIR}/${variant_stem}_spirv.hpp")

        add_custom_command(
            OUTPUT "${generated_header}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${VULKAN_GENERATED_INCLUDE_DIR}"
            COMMAND "${OV_GPU_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -I "${VULKAN_KERNEL_DIR}"
                    "-DELTWISE_FIXED_FUSED_CHAIN_LENGTH=${chain_length}" ${vector_definitions}
                    ${restrict_definition} "${shader_source}" -o "${raw_spirv}"
            COMMAND "${OV_GPU_SPIRV_OPT_EXECUTABLE}" -O "${raw_spirv}" -o "${optimized_spirv}"
            COMMAND "${OV_GPU_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${optimized_spirv}"
            COMMAND "${Python3_EXECUTABLE}" "${VULKAN_SHADER_GENERATOR}"
                    "${optimized_spirv}" "${generated_header}" "${variant_stem}_spirv"
                    "kernels/eltwise_fused_dense_chain.comp fixed chain length=${chain_length}${variant_description}"
            DEPENDS
                "${shader_source}"
                "${VULKAN_KERNEL_DIR}/eltwise_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc"
                "${VULKAN_SHADER_GENERATOR}"
            COMMENT "Generating host-compiled Vulkan shader ${variant_stem}"
            VERBATIM)
        list(APPEND VULKAN_GENERATED_HEADERS "${generated_header}")
    endforeach()
    set(VULKAN_GENERATED_HEADERS "${VULKAN_GENERATED_HEADERS}" PARENT_SCOPE)
endfunction()

# Read the shared host/shader resource limit instead of duplicating its value in CMake.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc" fused_chain_limit_definition
     REGEX "^ELTWISE_SHADER_LIMIT\\(max_fused_chain_length, [0-9]+\\)$")
string(REGEX REPLACE ".*, ([0-9]+)\\).*" "\\1" max_fused_chain_length "${fused_chain_limit_definition}")
set(min_multi_stage_fused_chain_length 2)
foreach(chain_length RANGE ${min_multi_stage_fused_chain_length} ${max_fused_chain_length})
    add_vulkan_fused_chain_shader(${chain_length} 0 FALSE)
    add_vulkan_fused_chain_shader(${chain_length} 4 TRUE)
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
