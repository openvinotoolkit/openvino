// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

uvec2 float_output_bits(float value, uint output_type) {
    if (output_type == type_f32) {
        return uvec2(floatBitsToUint(value), 0);
    }
    if (output_type == type_f16) {
        return uvec2(packHalf2x16(vec2(value, 0.0)), 0);
    }
    if (is_signed_type(output_type)) {
        int converted = int(value);
        return uvec2(uint(converted), converted < 0 ? 0xffffffff : 0);
    }
    return uvec2(uint(value), 0);
}
#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN || ELTWISE_FUSED_POST_OP
uvec2 evaluate_base_element(uint linear_index, out uint output_offset) {
#else
uvec2 evaluate_element(uint linear_index, out uint output_offset) {
#endif
    uint mode = selected_mode;
    uint input0_type = selected_input0_type;
#if !ELTWISE_UNARY
    uint input1_type = selected_input1_type;
#endif
    uint output_type = selected_base_output_type;
#if ELTWISE_SCALAR_CONSTANT
    uint scalar_input_index = scalar_constant_input_index();
    uint tensor_input_index = scalar_input_index == tensor_input0 ? tensor_input1 : tensor_input0;
    uint tensor_input_offset;
#else
    uint input0_offset;
#if !ELTWISE_UNARY
    uint input1_offset;
#endif
#endif
#if ELTWISE_DENSE
#if ELTWISE_DENSE_PUSH_CONSTANTS
    input0_offset = dense_metadata.input0_offset + linear_index;
    input1_offset = dense_metadata.input1_offset + linear_index;
    output_offset = dense_metadata.output_offset + linear_index;
#else
    input0_offset = metadata.values[header_words + tensor_input0 * tensor_words + max_rank * 2] + linear_index;
    input1_offset = metadata.values[header_words + tensor_input1 * tensor_words + max_rank * 2] + linear_index;
    output_offset = metadata.values[header_words + tensor_output * tensor_words + max_rank * 2] + linear_index;
#endif
#else
    if ((selected_storage_flags & storage_linear_flag) != 0) {
#if ELTWISE_SCALAR_CONSTANT
        tensor_input_offset = metadata.values[header_words + tensor_input_index * tensor_words + max_rank * 2] + linear_index;
#else
        input0_offset = metadata.values[header_words + tensor_input0 * tensor_words + max_rank * 2] + linear_index;
#if !ELTWISE_UNARY
        input1_offset = metadata.values[header_words + tensor_input1 * tensor_words + max_rank * 2] + linear_index;
#endif
#endif
        output_offset = metadata.values[header_words + tensor_output * tensor_words + max_rank * 2] + linear_index;
    } else {
#if ELTWISE_FAST_BROADCAST
        fast_broadcast_offsets(linear_index, input0_offset, input1_offset, output_offset);
#else
        uint rank = metadata.values[metadata_rank];
        uint output_base = header_words + tensor_output * tensor_words;
        uint coordinates[max_rank];
        uint remainder = linear_index;
        for (int axis = int(rank) - 1; axis >= 0; --axis) {
            uint dimension = metadata.values[output_base + uint(axis)];
            coordinates[axis] = remainder % dimension;
            remainder /= dimension;
        }
#if ELTWISE_SCALAR_CONSTANT
        tensor_input_offset = storage_element_offset(tensor_input_index, coordinates, rank);
#else
        input0_offset = storage_element_offset(tensor_input0, coordinates, rank);
#if !ELTWISE_UNARY
        input1_offset = storage_element_offset(tensor_input1, coordinates, rank);
#endif
#endif
        output_offset = storage_element_offset(tensor_output, coordinates, rank);
#endif
    }
#endif

#if ELTWISE_SCALAR_CONSTANT
    if (is_float_type(input0_type)) {
        float lhs = scalar_input_index == tensor_input0 ? scalar_constant_float(input0_type) : load_float_0(tensor_input_offset, input0_type);
        float rhs = scalar_input_index == tensor_input1 ? scalar_constant_float(input1_type) : load_float_1(tensor_input_offset, input1_type);
        if (is_boolean_mode(mode)) {
            return uvec2(apply_float_boolean(lhs, rhs, mode) ? 1 : 0, 0);
        }
        return float_output_bits(apply_float(lhs, rhs, mode), output_type);
    } else {
        uvec2 lhs = scalar_input_index == tensor_input0 ? scalar_constant_bits(input0_type) : load_integer_0(tensor_input_offset, input0_type);
        uvec2 rhs = scalar_input_index == tensor_input1 ? scalar_constant_bits(input1_type) : load_integer_1(tensor_input_offset, input1_type);
        bool signed_type = is_signed_type(input0_type);
        if (is_boolean_mode(mode)) {
            return uvec2(apply_integer_boolean(lhs, rhs, mode, signed_type) ? 1 : 0, 0);
        }
        return apply_integer(lhs, rhs, mode, signed_type);
    }
#else
    if (is_float_type(input0_type)) {
        float lhs = load_float_0(input0_offset, input0_type);
#if ELTWISE_UNARY
        return uvec2(apply_float_unary_boolean(lhs, mode) ? 1 : 0, 0);
#else
        float rhs = load_float_1(input1_offset, input1_type);
        if (is_boolean_mode(mode)) {
            return uvec2(apply_float_boolean(lhs, rhs, mode) ? 1 : 0, 0);
        }
        return float_output_bits(apply_float(lhs, rhs, mode), output_type);
#endif
    } else {
#if ELTWISE_UNARY
        return uvec2(0);
#else
        uvec2 lhs = load_integer_0(input0_offset, input0_type);
        uvec2 rhs = load_integer_1(input1_offset, input1_type);
        bool signed_type = is_signed_type(input0_type);
        if (is_boolean_mode(mode)) {
            return uvec2(apply_integer_boolean(lhs, rhs, mode, signed_type) ? 1 : 0, 0);
        }
        return apply_integer(lhs, rhs, mode, signed_type);
#endif
    }
#endif
}
