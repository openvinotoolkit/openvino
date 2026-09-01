// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

float post_op_parameter(uint field) {
    return uintBitsToFloat(metadata.values[field]);
}

float post_op_input_value(uvec2 value) {
    return selected_base_output_type == type_f32 ? uintBitsToFloat(value.x) : unpackHalf2x16(value.x).x;
}

float apply_post_activation(float value) {
    float parameter_a = post_op_parameter(post_op_metadata_parameter_a);
    float parameter_b = post_op_parameter(post_op_metadata_parameter_b);
    switch (selected_post_activation) {
    case post_activation_none:
        return value;
    case post_activation_relu:
        return max(value, 0.0);
    case post_activation_relu_negative_slope:
        if (isinf(parameter_a)) {
            return value >= 0.0 ? value : -parameter_a;
        }
        return max(value, 0.0) + parameter_a * min(value, 0.0);
    case post_activation_clamp:
        return max(parameter_a, min(parameter_b, value));
    case post_activation_abs:
        return abs(value);
    case post_activation_linear:
        return parameter_a * value + parameter_b;
    case post_activation_square:
        return value * value;
    case post_activation_sqrt:
        return sqrt(value);
    case post_activation_negative:
        return -value;
    case post_activation_floor:
        return floor(value);
    case post_activation_ceil:
        return ceil(value);
    case post_activation_reciprocal:
        return 1.0 / value;
    case post_activation_hard_sigmoid:
        return clamp(parameter_a * value + parameter_b, 0.0, 1.0);
    case post_activation_hsigmoid:
        return min(max(0.0, value + 3.0), 6.0) / 6.0;
    case post_activation_hswish:
        return value * min(max(0.0, value + 3.0), 6.0) / 6.0;
    default:
        return value;
    }
}

bool post_quantize_flag(uint flag) {
    return (selected_post_quantize_flags & flag) != 0;
}

float round_half_away_from_zero(float value) {
    return value >= 0.0 ? floor(value + 0.5) : ceil(value - 0.5);
}

float apply_post_quantize(float value) {
    float input_low = post_op_parameter(post_op_metadata_input_low);
    float input_high = post_op_parameter(post_op_metadata_input_high);
    float input_scale = post_op_parameter(post_op_metadata_input_scale);
    float input_shift = post_op_parameter(post_op_metadata_input_shift);
    float output_low = post_op_parameter(post_op_metadata_output_low);
    float output_high = post_op_parameter(post_op_metadata_output_high);
    float output_scale = post_op_parameter(post_op_metadata_output_scale);
    float output_shift = post_op_parameter(post_op_metadata_output_shift);
    bool use_output_range = post_quantize_flag(post_quantize_use_output_range_flag);
    bool need_clamp = post_quantize_flag(post_quantize_need_clamp_flag);

    if (!use_output_range && need_clamp) {
        if (post_quantize_flag(post_quantize_need_min_clamp_flag)) {
            value = max(value, input_low);
        }
        if (post_quantize_flag(post_quantize_need_max_clamp_flag)) {
            value = min(value, input_high);
        }
    }

    value *= input_scale;
    if (post_quantize_flag(post_quantize_need_pre_shift_flag)) {
        value += input_shift;
    }

    bool output_is_i8_u8 = selected_output_type == type_i8 || selected_output_type == type_u8;
    bool has_post_transform = post_quantize_flag(post_quantize_need_post_scale_flag) ||
                              post_quantize_flag(post_quantize_need_post_shift_flag);
    if (!output_is_i8_u8 || has_post_transform) {
        value = round_half_away_from_zero(value);
    }
    if (post_quantize_flag(post_quantize_need_post_scale_flag)) {
        value *= output_scale;
    }
    if (post_quantize_flag(post_quantize_need_post_shift_flag)) {
        value += output_shift;
    }

    if (use_output_range && need_clamp) {
        if (post_quantize_flag(post_quantize_need_min_clamp_flag)) {
            value = max(value, output_low);
        }
        if (post_quantize_flag(post_quantize_need_max_clamp_flag)) {
            value = min(value, output_high);
        }
    }
    if (selected_output_type == type_i8) {
        value = roundEven(clamp(value, -128.0, 127.0));
    } else if (selected_output_type == type_u8) {
        value = roundEven(clamp(value, 0.0, 255.0));
    }
    return value;
}

uvec2 evaluate_element(uint linear_index, out uint output_offset) {
    uvec2 base_value = evaluate_base_element(linear_index, output_offset);
    float value = post_op_input_value(base_value);
    if (selected_post_op_kind == post_op_activation) {
        value = apply_post_activation(value);
    } else if (selected_post_op_kind == post_op_quantize) {
        value = apply_post_quantize(value);
    }
    return float_output_bits(value, selected_output_type);
}
