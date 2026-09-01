// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

uvec2 evaluate_element(uint linear_index, out uint output_offset) {
    uvec2 base_value = evaluate_base_element(linear_index, output_offset);
    uint output_type = selected_output_type;
#if ELTWISE_FUSED_BROADCAST
    uint fused_input_offset = fused_broadcast_input_offset(linear_index);
#else
    uint fused_input_offset = runtime_fused_input_offset() + linear_index;
#endif
    if (is_float_type(output_type)) {
        float original = output_type == type_f32 ? uintBitsToFloat(base_value.x) : unpackHalf2x16(base_value.x).x;
        float fused_input = load_float_fused(fused_input_offset, selected_fused_input_type);
        float lhs = selected_fused_input_position == fused_input_lhs ? fused_input : original;
        float rhs = selected_fused_input_position == fused_input_rhs ? fused_input : original;
        return float_output_bits(apply_fused_float(lhs, rhs), output_type);
    }

    uvec2 fused_input = load_integer_fused(fused_input_offset, selected_fused_input_type);
    uvec2 lhs = selected_fused_input_position == fused_input_lhs ? fused_input : base_value;
    uvec2 rhs = selected_fused_input_position == fused_input_rhs ? fused_input : base_value;
    return apply_fused_integer(lhs, rhs, is_signed_type(output_type));
}
