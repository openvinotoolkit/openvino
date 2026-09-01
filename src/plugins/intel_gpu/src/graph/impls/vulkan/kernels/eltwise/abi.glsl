// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define ELTWISE_SHADER_MODE(name, code) const uint mode_##name = code;
#define ELTWISE_SHADER_SCALAR_TYPE(name, code) const uint type_##name = code;
#define ELTWISE_SHADER_METADATA_FIELD(name, code) const uint metadata_##name = code;
#define ELTWISE_SHADER_TENSOR_INDEX(name, code) const uint tensor_##name = code;
#define ELTWISE_SHADER_STORAGE_FLAG(name, code) const uint storage_##name##_flag = code;
#define ELTWISE_SHADER_INFINITY_FLAG(name, code) const uint infinity_##name##_flag = code;
#define ELTWISE_SHADER_POST_OP_KIND(name, code) const uint post_op_##name = code;
#define ELTWISE_SHADER_POST_ACTIVATION(name, code) const uint post_activation_##name = code;
#define ELTWISE_SHADER_POST_QUANTIZE_FLAG(name, code) const uint post_quantize_##name##_flag = code;
#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
#define ELTWISE_SHADER_FUSED_INPUT_POSITION(name, code) const uint fused_input_##name = code;
#endif
#define ELTWISE_SHADER_LIMIT(name, code) const uint name = code;
#define ELTWISE_SHADER_SPECIALIZATION_ID(name, code) const uint specialization_##name##_id = code;
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_SPECIALIZATION_ID
#undef ELTWISE_SHADER_LIMIT
#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
#undef ELTWISE_SHADER_FUSED_INPUT_POSITION
#endif
#undef ELTWISE_SHADER_INFINITY_FLAG
#undef ELTWISE_SHADER_POST_QUANTIZE_FLAG
#undef ELTWISE_SHADER_POST_ACTIVATION
#undef ELTWISE_SHADER_POST_OP_KIND
#undef ELTWISE_SHADER_STORAGE_FLAG
#undef ELTWISE_SHADER_TENSOR_INDEX
#undef ELTWISE_SHADER_METADATA_FIELD
#undef ELTWISE_SHADER_SCALAR_TYPE
#undef ELTWISE_SHADER_MODE

layout(local_size_x_id = specialization_local_size_x_id, local_size_y = 1, local_size_z = 1) in;
#ifdef ELTWISE_FIXED_MODE
const uint selected_mode = uint(ELTWISE_FIXED_MODE);
#else
layout(constant_id = specialization_mode_id) const uint selected_mode = mode_sum;
#endif
#ifdef ELTWISE_FIXED_INPUT0_TYPE
const uint selected_input0_type = uint(ELTWISE_FIXED_INPUT0_TYPE);
#else
layout(constant_id = specialization_input0_type_id) const uint selected_input0_type = type_f32;
#endif
#ifdef ELTWISE_FIXED_INPUT1_TYPE
const uint selected_input1_type = uint(ELTWISE_FIXED_INPUT1_TYPE);
#else
layout(constant_id = specialization_input1_type_id) const uint selected_input1_type = type_f32;
#endif
#ifdef ELTWISE_FIXED_OUTPUT_TYPE
const uint selected_output_type = uint(ELTWISE_FIXED_OUTPUT_TYPE);
#else
layout(constant_id = specialization_output_type_id) const uint selected_output_type = type_f32;
#endif
#if ELTWISE_FUSED_POST_OP
layout(constant_id = specialization_base_output_type_id) const uint selected_base_output_type = type_f32;
layout(constant_id = specialization_post_op_kind_id) const uint selected_post_op_kind = post_op_activation;
layout(constant_id = specialization_post_activation_id) const uint selected_post_activation = post_activation_none;
layout(constant_id = specialization_post_quantize_flags_id) const uint selected_post_quantize_flags = 0;
#else
const uint selected_base_output_type = selected_output_type;
#endif
layout(constant_id = specialization_storage_flags_id) const uint selected_storage_flags = 0;
#if ELTWISE_DENSE || ELTWISE_BROADCAST_VECTOR || ELTWISE_SCALAR_CONSTANT
layout(constant_id = specialization_elements_per_invocation_id) const uint selected_elements_per_invocation = 1;
#endif
#if ELTWISE_FUSED
layout(constant_id = specialization_fused_mode_id) const uint selected_fused_mode = mode_sum;
layout(constant_id = specialization_fused_input_type_id) const uint selected_fused_input_type = type_f32;
layout(constant_id = specialization_fused_input_position_id) const uint selected_fused_input_position = fused_input_rhs;
#endif
#if ELTWISE_FUSED_CHAIN && ELTWISE_FIXED_FUSED_CHAIN_LENGTH > 0
#define ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(stage)                                                                      \
    layout(constant_id = specialization_fused_chain_mode_base_id + stage) const uint selected_fused_chain_mode_##stage =       \
        mode_sum;                                                                                                              \
    layout(constant_id = specialization_fused_chain_input_type_base_id + stage) const uint                                    \
        selected_fused_chain_input_type_##stage = type_f32;                                                                    \
    layout(constant_id = specialization_fused_chain_input_position_base_id + stage) const uint                                \
        selected_fused_chain_input_position_##stage = fused_input_rhs;
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(0)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(1)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(2)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(3)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(4)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(5)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(6)
ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION(7)
#undef ELTWISE_DECLARE_FUSED_CHAIN_SPECIALIZATION
#endif
#if ELTWISE_FAST_BROADCAST
layout(constant_id = specialization_broadcast_rank_id) const uint selected_broadcast_rank = 1;
layout(constant_id = specialization_broadcast_input0_axes_id) const uint selected_broadcast_input0_axes = 0;
layout(constant_id = specialization_broadcast_input1_axes_id) const uint selected_broadcast_input1_axes = 0;
layout(constant_id = specialization_broadcast_output_axes_id) const uint selected_broadcast_output_axes = 0;
#endif

const uint max_rank = 8;
const uint header_words = metadata_count;
const uint tensor_words = max_rank * 2 + 1;
const uint fused_metadata_base = header_words + tensor_count * tensor_words;
#define ELTWISE_SHADER_FUSED_METADATA_FIELD(name, code) const uint fused_metadata_##name = code;
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_METADATA_FIELD
#define ELTWISE_SHADER_FUSED_METADATA_FIELD(name, code) const uint metadata_fused_##name = fused_metadata_base + fused_metadata_##name;
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_METADATA_FIELD
const uint fused_chain_metadata_base = fused_metadata_base + max_fused_chain_length * fused_metadata_count;
#define ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD(name, code) const uint fused_chain_metadata_##name = code;
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD
#define ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD(name, code) const uint metadata_fused_chain_##name = fused_chain_metadata_base + fused_chain_metadata_##name;
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD
#if ELTWISE_FUSED_CHAIN
const uint fast_divisor_metadata_base = fused_chain_metadata_base + fused_chain_metadata_count;
#else
const uint fast_divisor_metadata_base = fused_metadata_base + fused_metadata_count;
#endif
const uint fused_broadcast_metadata_base = fused_chain_metadata_base + fused_chain_metadata_count + max_rank;
const uint post_op_metadata_base = fused_broadcast_metadata_base + tensor_words;
#define ELTWISE_SHADER_POST_OP_METADATA_FIELD(name, code) const uint post_op_metadata_##name = post_op_metadata_base + code;
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_POST_OP_METADATA_FIELD
