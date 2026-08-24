# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(ELTWISE_SHADER_ENTRY "${VULKAN_ELTWISE_KERNEL_DIR}/entry.comp")
set(ELTWISE_SHADER_MODULES
    "${VULKAN_ELTWISE_KERNEL_DIR}/program.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/configuration.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/bindings.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/abi.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/metadata.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/storage.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/integer_math.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/broadcasting.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/operations.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/evaluation.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/evaluation_base.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/post_operations.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/fused_evaluation.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/fused_chain_evaluation.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/output.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/dispatch.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/dispatch_packed.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/dispatch_f32_vector.glsl"
    "${VULKAN_ELTWISE_KERNEL_DIR}/dispatch_scalar.glsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_shader_abi.inc"
)

function(add_eltwise_shader_variant)
    set(options
        DENSE
        BROADCAST_VECTOR
        FAST_BROADCAST
        UNARY
        SCALAR_CONSTANT
        FUSED
        FUSED_BROADCAST
        FUSED_CHAIN
        FUSED_POST_OP
        DENSE_PUSH_CONSTANTS
        F32_NO_TAIL
        PACKED_FLOAT16
        ALLOW_RESTRICT
    )
    set(one_value_args
        NAME
        F32_VECTOR_WIDTH
        PACKED_VECTOR_WIDTH
        OPTIMIZATION
        DESCRIPTION
    )
    cmake_parse_arguments(PARSE_ARGV 0 ELTWISE "${options}" "${one_value_args}" "DEFINITIONS")

    if(NOT ELTWISE_NAME)
        message(FATAL_ERROR "An Eltwise shader variant requires NAME")
    endif()

    set(definitions)
    foreach(feature IN ITEMS DENSE BROADCAST_VECTOR UNARY SCALAR_CONSTANT)
        set(value 0)
        if(ELTWISE_${feature})
            set(value 1)
        endif()
        list(APPEND definitions "ELTWISE_${feature}=${value}")
    endforeach()

    foreach(feature IN ITEMS
            FAST_BROADCAST
            FUSED
            FUSED_BROADCAST
            FUSED_CHAIN
            FUSED_POST_OP
            DENSE_PUSH_CONSTANTS
            F32_NO_TAIL
            PACKED_FLOAT16)
        if(ELTWISE_${feature})
            list(APPEND definitions "ELTWISE_${feature}=1")
        endif()
    endforeach()

    if(ELTWISE_F32_VECTOR_WIDTH)
        list(APPEND definitions "ELTWISE_F32_VECTOR_WIDTH=${ELTWISE_F32_VECTOR_WIDTH}")
    endif()
    if(ELTWISE_PACKED_VECTOR_WIDTH)
        list(APPEND definitions "ELTWISE_PACKED_VECTOR_WIDTH=${ELTWISE_PACKED_VECTOR_WIDTH}")
    endif()
    list(APPEND definitions ${ELTWISE_DEFINITIONS})

    if(ELTWISE_DESCRIPTION)
        set(description "${ELTWISE_DESCRIPTION}")
    else()
        set(description "Eltwise variant ${ELTWISE_NAME}")
    endif()

    add_vulkan_shader(
        NAME "${ELTWISE_NAME}"
        SOURCE "${ELTWISE_SHADER_ENTRY}"
        OPTIMIZATION "${ELTWISE_OPTIMIZATION}"
        DESCRIPTION "${description}"
        DEFINITIONS ${definitions}
        DEPENDENCIES ${ELTWISE_SHADER_MODULES}
    )

    if(ELTWISE_ALLOW_RESTRICT)
        add_vulkan_shader(
            NAME "${ELTWISE_NAME}_restrict"
            SOURCE "${ELTWISE_SHADER_ENTRY}"
            OPTIMIZATION "${ELTWISE_OPTIMIZATION}"
            DESCRIPTION "${description} with runtime-proven non-aliasing output"
            DEFINITIONS ${definitions} ELTWISE_RESTRICT_OUTPUT=1
            DEPENDENCIES ${ELTWISE_SHADER_MODULES}
        )
    endif()
endfunction()

# General layouts and metadata paths.
add_eltwise_shader_variant(NAME eltwise)
add_eltwise_shader_variant(NAME eltwise_broadcast_fast FAST_BROADCAST)
add_eltwise_shader_variant(NAME eltwise_broadcast_vector BROADCAST_VECTOR)
add_eltwise_shader_variant(NAME eltwise_broadcast_fast_vector BROADCAST_VECTOR FAST_BROADCAST)
add_eltwise_shader_variant(NAME eltwise_unary UNARY)
add_eltwise_shader_variant(NAME eltwise_scalar_constant SCALAR_CONSTANT)

# Dense scalar, vector, and packed paths.
add_eltwise_shader_variant(NAME eltwise_dense DENSE ALLOW_RESTRICT)
add_eltwise_shader_variant(
    NAME eltwise_dense_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT)
add_eltwise_shader_variant(
    NAME eltwise_dense_f32_vec2_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    F32_VECTOR_WIDTH 2)
add_eltwise_shader_variant(
    NAME eltwise_dense_f32_vec2_no_tail_push_constants
    DENSE DENSE_PUSH_CONSTANTS F32_NO_TAIL ALLOW_RESTRICT
    F32_VECTOR_WIDTH 2)
add_eltwise_shader_variant(
    NAME eltwise_dense_f32_vec4_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    F32_VECTOR_WIDTH 4)
add_eltwise_shader_variant(
    NAME eltwise_dense_f32_vec4_no_tail_push_constants
    DENSE DENSE_PUSH_CONSTANTS F32_NO_TAIL ALLOW_RESTRICT
    F32_VECTOR_WIDTH 4)
add_eltwise_shader_variant(
    NAME eltwise_dense_packed_16bit_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    PACKED_VECTOR_WIDTH 2)
add_eltwise_shader_variant(
    NAME eltwise_dense_packed_8bit_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    PACKED_VECTOR_WIDTH 4)
add_eltwise_shader_variant(
    NAME eltwise_dense_packed_f16_push_constants
    DENSE DENSE_PUSH_CONSTANTS PACKED_FLOAT16 ALLOW_RESTRICT
    PACKED_VECTOR_WIDTH 2)

# Fused paths reuse the same modules and add only their execution contract.
add_eltwise_shader_variant(NAME eltwise_fused_broadcast DENSE FUSED FUSED_BROADCAST)
add_eltwise_shader_variant(NAME eltwise_fused_dense DENSE FUSED ALLOW_RESTRICT)
add_eltwise_shader_variant(NAME eltwise_fused_dense_chain DENSE FUSED_CHAIN ALLOW_RESTRICT)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS ALLOW_RESTRICT)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_f32_vec2_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    F32_VECTOR_WIDTH 2)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_f32_vec2_no_tail_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS F32_NO_TAIL ALLOW_RESTRICT
    F32_VECTOR_WIDTH 2)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_f32_vec4_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    F32_VECTOR_WIDTH 4)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_f32_vec4_no_tail_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS F32_NO_TAIL ALLOW_RESTRICT
    F32_VECTOR_WIDTH 4)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_packed_16bit_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    PACKED_VECTOR_WIDTH 2)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_packed_8bit_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS ALLOW_RESTRICT
    PACKED_VECTOR_WIDTH 4)
add_eltwise_shader_variant(
    NAME eltwise_fused_dense_packed_f16_push_constants
    DENSE FUSED DENSE_PUSH_CONSTANTS PACKED_FLOAT16 ALLOW_RESTRICT
    PACKED_VECTOR_WIDTH 2)
add_eltwise_shader_variant(NAME eltwise_fused_post_op DENSE FUSED_POST_OP)

# Bounded specializations for common arithmetic and comparison modes.
add_eltwise_shader_variant(
    NAME eltwise_dense_f32_sum_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT OPTIMIZATION "-O"
    DEFINITIONS ELTWISE_FIXED_MODE=mode_sum ELTWISE_FIXED_INPUT0_TYPE=type_f32
                ELTWISE_FIXED_INPUT1_TYPE=type_f32 ELTWISE_FIXED_OUTPUT_TYPE=type_f32)
add_eltwise_shader_variant(
    NAME eltwise_dense_f32_div_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT OPTIMIZATION "-O"
    DEFINITIONS ELTWISE_FIXED_MODE=mode_div ELTWISE_FIXED_INPUT0_TYPE=type_f32
                ELTWISE_FIXED_INPUT1_TYPE=type_f32 ELTWISE_FIXED_OUTPUT_TYPE=type_f32)
add_eltwise_shader_variant(
    NAME eltwise_dense_i64_sum_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT OPTIMIZATION "-O"
    DEFINITIONS ELTWISE_FIXED_MODE=mode_sum ELTWISE_FIXED_INPUT0_TYPE=type_i64
                ELTWISE_FIXED_INPUT1_TYPE=type_i64 ELTWISE_FIXED_OUTPUT_TYPE=type_i64)
add_eltwise_shader_variant(
    NAME eltwise_dense_i64_div_push_constants
    DENSE DENSE_PUSH_CONSTANTS ALLOW_RESTRICT OPTIMIZATION "-O"
    DEFINITIONS ELTWISE_FIXED_MODE=mode_div ELTWISE_FIXED_INPUT0_TYPE=type_i64
                ELTWISE_FIXED_INPUT1_TYPE=type_i64 ELTWISE_FIXED_OUTPUT_TYPE=type_i64)
add_eltwise_shader_variant(
    NAME eltwise_broadcast_f32_eq OPTIMIZATION "-O"
    DEFINITIONS ELTWISE_FIXED_MODE=mode_eq ELTWISE_FIXED_INPUT0_TYPE=type_f32
                ELTWISE_FIXED_INPUT1_TYPE=type_f32 ELTWISE_FIXED_OUTPUT_TYPE=type_boolean)
add_eltwise_shader_variant(
    NAME eltwise_broadcast_fast_f32_eq FAST_BROADCAST OPTIMIZATION "-O"
    DEFINITIONS ELTWISE_FIXED_MODE=mode_eq ELTWISE_FIXED_INPUT0_TYPE=type_f32
                ELTWISE_FIXED_INPUT1_TYPE=type_f32 ELTWISE_FIXED_OUTPUT_TYPE=type_boolean)

# Fixed fused-chain lengths let the host select an unrolled scalar or vec4 path.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/eltwise_shader_abi.inc" fused_chain_limit_definition
     REGEX "^ELTWISE_SHADER_LIMIT\\(max_fused_chain_length, [0-9]+\\)$")
string(REGEX REPLACE ".*, ([0-9]+)\\).*" "\\1" max_fused_chain_length "${fused_chain_limit_definition}")
set(min_multi_stage_fused_chain_length 2)
foreach(chain_length RANGE ${min_multi_stage_fused_chain_length} ${max_fused_chain_length})
    add_eltwise_shader_variant(
        NAME "eltwise_fused_dense_chain_length${chain_length}"
        DENSE FUSED_CHAIN ALLOW_RESTRICT OPTIMIZATION "-O"
        DEFINITIONS "ELTWISE_FIXED_FUSED_CHAIN_LENGTH=${chain_length}")
    add_eltwise_shader_variant(
        NAME "eltwise_fused_dense_chain_length${chain_length}_f32_vec4_no_tail"
        DENSE FUSED_CHAIN F32_NO_TAIL ALLOW_RESTRICT OPTIMIZATION "-O"
        F32_VECTOR_WIDTH 4
        DEFINITIONS "ELTWISE_FIXED_FUSED_CHAIN_LENGTH=${chain_length}")
endforeach()
