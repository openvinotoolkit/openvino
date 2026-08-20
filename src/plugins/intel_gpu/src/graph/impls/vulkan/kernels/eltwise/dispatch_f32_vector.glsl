// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if ELTWISE_F32_VECTOR_WIDTH == 2
#define ELTWISE_FLOAT_VECTOR vec2
#define ELTWISE_UINT_VECTOR uvec2
#define ELTWISE_BOOL_VECTOR bvec2
#else
#define ELTWISE_FLOAT_VECTOR vec4
#define ELTWISE_UINT_VECTOR uvec4
#define ELTWISE_BOOL_VECTOR bvec4
#endif

ELTWISE_BOOL_VECTOR logical_or_vector(ELTWISE_BOOL_VECTOR lhs, ELTWISE_BOOL_VECTOR rhs) {
#if ELTWISE_F32_VECTOR_WIDTH == 2
    return ELTWISE_BOOL_VECTOR(lhs.x || rhs.x, lhs.y || rhs.y);
#else
    return ELTWISE_BOOL_VECTOR(lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z, lhs.w || rhs.w);
#endif
}

ELTWISE_FLOAT_VECTOR truncate_toward_zero_vector(ELTWISE_FLOAT_VECTOR value) {
    return mix(floor(value), ceil(value), lessThan(value, ELTWISE_FLOAT_VECTOR(0.0)));
}

ELTWISE_FLOAT_VECTOR apply_mod_vector(ELTWISE_FLOAT_VECTOR lhs, ELTWISE_FLOAT_VECTOR rhs) {
    ELTWISE_FLOAT_VECTOR result = lhs - truncate_toward_zero_vector(lhs / rhs) * rhs;
    result = mix(result, lhs, logical_or_vector(equal(lhs, ELTWISE_FLOAT_VECTOR(0.0)), isinf(rhs)));
    result = mix(result, lhs * ELTWISE_FLOAT_VECTOR(0.0), equal(abs(lhs), abs(rhs)));
    return mix(result,
               ELTWISE_FLOAT_VECTOR(quiet_nan()),
               logical_or_vector(equal(rhs, ELTWISE_FLOAT_VECTOR(0.0)), isinf(lhs)));
}

ELTWISE_FLOAT_VECTOR apply_pow_vector(ELTWISE_FLOAT_VECTOR lhs, ELTWISE_FLOAT_VECTOR rhs) {
    ELTWISE_FLOAT_VECTOR positive_result = pow(lhs, rhs);
    ELTWISE_FLOAT_VECTOR integer_exponent = truncate_toward_zero_vector(rhs);
    ELTWISE_FLOAT_VECTOR magnitude = pow(-lhs, rhs);
    ELTWISE_FLOAT_VECTOR exponent_pairs = truncate_toward_zero_vector(integer_exponent * ELTWISE_FLOAT_VECTOR(0.5));
    ELTWISE_BOOL_VECTOR odd_exponent = notEqual(integer_exponent, exponent_pairs * ELTWISE_FLOAT_VECTOR(2.0));
    ELTWISE_FLOAT_VECTOR negative_result = mix(magnitude, -magnitude, odd_exponent);
    negative_result = mix(ELTWISE_FLOAT_VECTOR(quiet_nan()), negative_result, equal(integer_exponent, rhs));
    ELTWISE_FLOAT_VECTOR result = mix(negative_result, positive_result, greaterThanEqual(lhs, ELTWISE_FLOAT_VECTOR(0.0)));
    return mix(result, ELTWISE_FLOAT_VECTOR(1.0), equal(rhs, ELTWISE_FLOAT_VECTOR(0.0)));
}

ELTWISE_FLOAT_VECTOR apply_float_vector(ELTWISE_FLOAT_VECTOR lhs, ELTWISE_FLOAT_VECTOR rhs, uint mode) {
    switch (mode) {
    case mode_sum:
        return lhs * ELTWISE_FLOAT_VECTOR(uintBitsToFloat(runtime_input0_coefficient())) +
               rhs * ELTWISE_FLOAT_VECTOR(uintBitsToFloat(runtime_input1_coefficient()));
    case mode_sub:
        return lhs - rhs;
    case mode_max:
        return max(lhs, rhs);
    case mode_prod:
        return lhs * rhs;
    case mode_div:
        return lhs / rhs;
    case mode_min:
        return min(lhs, rhs);
    case mode_pow:
        return apply_pow_vector(lhs, rhs);
    case mode_squared_diff: {
        ELTWISE_FLOAT_VECTOR difference = lhs - rhs;
        return difference * difference;
    }
    case mode_mod:
        return apply_mod_vector(lhs, rhs);
    case mode_floor_mod: {
        ELTWISE_FLOAT_VECTOR result = lhs - floor(lhs / rhs) * rhs;
        return mix(result,
                   ELTWISE_FLOAT_VECTOR(0.0),
                   logical_or_vector(equal(rhs, ELTWISE_FLOAT_VECTOR(0.0)), equal(lhs, rhs)));
    }
    case mode_atan2:
        return atan(lhs, rhs);
    default:
        return ELTWISE_FLOAT_VECTOR(0.0);
    }
}

#if ELTWISE_F32_VECTOR_WIDTH == 2
ELTWISE_FLOAT_VECTOR load_f32_vector_0(uint offset) {
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(input0_data.values[offset], input0_data.values[offset + 1]));
}

ELTWISE_FLOAT_VECTOR load_f32_vector_1(uint offset) {
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(input1_data.values[offset], input1_data.values[offset + 1]));
}

#if ELTWISE_FUSED
ELTWISE_FLOAT_VECTOR load_f32_vector_fused(uint offset) {
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(fused_input_data.values[offset], fused_input_data.values[offset + 1]));
}
#endif

void store_f32_vector(uint offset, ELTWISE_FLOAT_VECTOR value) {
    ELTWISE_UINT_VECTOR bits = floatBitsToUint(value);
    output_data.values[offset] = bits.x;
    output_data.values[offset + 1] = bits.y;
}
#else
ELTWISE_FLOAT_VECTOR load_f32_vector_0(uint offset) {
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(input0_data.values[offset],
                                              input0_data.values[offset + 1],
                                              input0_data.values[offset + 2],
                                              input0_data.values[offset + 3]));
}

ELTWISE_FLOAT_VECTOR load_f32_vector_1(uint offset) {
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(input1_data.values[offset],
                                              input1_data.values[offset + 1],
                                              input1_data.values[offset + 2],
                                              input1_data.values[offset + 3]));
}

#if ELTWISE_FUSED
ELTWISE_FLOAT_VECTOR load_f32_vector_fused(uint offset) {
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(fused_input_data.values[offset],
                                              fused_input_data.values[offset + 1],
                                              fused_input_data.values[offset + 2],
                                              fused_input_data.values[offset + 3]));
}
#endif

void store_f32_vector(uint offset, ELTWISE_FLOAT_VECTOR value) {
    ELTWISE_UINT_VECTOR bits = floatBitsToUint(value);
    output_data.values[offset] = bits.x;
    output_data.values[offset + 1] = bits.y;
    output_data.values[offset + 2] = bits.z;
    output_data.values[offset + 3] = bits.w;
}
#endif

#if ELTWISE_FUSED_CHAIN
ELTWISE_FLOAT_VECTOR load_f32_vector_fused_chain(uint stage, uint offset) {
#if ELTWISE_F32_VECTOR_WIDTH == 2
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(load_u32_fused_chain(stage, offset * 4),
                                              load_u32_fused_chain(stage, (offset + 1) * 4)));
#else
    return uintBitsToFloat(ELTWISE_UINT_VECTOR(load_u32_fused_chain(stage, offset * 4),
                                              load_u32_fused_chain(stage, (offset + 1) * 4),
                                              load_u32_fused_chain(stage, (offset + 2) * 4),
                                              load_u32_fused_chain(stage, (offset + 3) * 4)));
#endif
}
#endif

#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
ELTWISE_FLOAT_VECTOR apply_fused_float_vector_runtime(ELTWISE_FLOAT_VECTOR lhs,
                                                      ELTWISE_FLOAT_VECTOR rhs,
                                                      uint mode,
                                                      uint input0_coefficient,
                                                      uint input1_coefficient) {
    if (mode == mode_sum) {
        return lhs * ELTWISE_FLOAT_VECTOR(uintBitsToFloat(input0_coefficient)) +
               rhs * ELTWISE_FLOAT_VECTOR(uintBitsToFloat(input1_coefficient));
    }
    return apply_float_vector(lhs, rhs, mode);
}
#endif

#if ELTWISE_FUSED
ELTWISE_FLOAT_VECTOR apply_fused_float_vector(ELTWISE_FLOAT_VECTOR lhs, ELTWISE_FLOAT_VECTOR rhs) {
    return apply_fused_float_vector_runtime(lhs,
                                            rhs,
                                            selected_fused_mode,
                                            runtime_fused_input0_coefficient(),
                                            runtime_fused_input1_coefficient());
}
#endif

#if ELTWISE_FUSED_CHAIN
ELTWISE_FLOAT_VECTOR evaluate_fused_chain_float_vector_stage(ELTWISE_FLOAT_VECTOR result,
                                                             uint stage,
                                                             uint first_element,
                                                             uint mode,
                                                             uint input_position) {
    uint input_offset = fused_chain_metadata(stage, fused_metadata_input_offset) + first_element;
    ELTWISE_FLOAT_VECTOR external_value = load_f32_vector_fused_chain(stage, input_offset);
    ELTWISE_FLOAT_VECTOR lhs = input_position == fused_input_lhs ? external_value : result;
    ELTWISE_FLOAT_VECTOR rhs = input_position == fused_input_rhs ? external_value : result;
    return apply_fused_float_vector_runtime(lhs,
                                            rhs,
                                            mode,
                                            fused_chain_metadata(stage, fused_metadata_input0_coefficient),
                                            fused_chain_metadata(stage, fused_metadata_input1_coefficient));
}

ELTWISE_FLOAT_VECTOR evaluate_fused_chain_float_vector(ELTWISE_FLOAT_VECTOR result, uint first_element) {
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 1
    result = evaluate_fused_chain_float_vector_stage(
        result, 0, first_element, selected_fused_chain_mode_0, selected_fused_chain_input_position_0);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 2
    result = evaluate_fused_chain_float_vector_stage(
        result, 1, first_element, selected_fused_chain_mode_1, selected_fused_chain_input_position_1);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 3
    result = evaluate_fused_chain_float_vector_stage(
        result, 2, first_element, selected_fused_chain_mode_2, selected_fused_chain_input_position_2);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 4
    result = evaluate_fused_chain_float_vector_stage(
        result, 3, first_element, selected_fused_chain_mode_3, selected_fused_chain_input_position_3);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 5
    result = evaluate_fused_chain_float_vector_stage(
        result, 4, first_element, selected_fused_chain_mode_4, selected_fused_chain_input_position_4);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 6
    result = evaluate_fused_chain_float_vector_stage(
        result, 5, first_element, selected_fused_chain_mode_5, selected_fused_chain_input_position_5);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 7
    result = evaluate_fused_chain_float_vector_stage(
        result, 6, first_element, selected_fused_chain_mode_6, selected_fused_chain_input_position_6);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 8
    result = evaluate_fused_chain_float_vector_stage(
        result, 7, first_element, selected_fused_chain_mode_7, selected_fused_chain_input_position_7);
#endif
    return result;
}
#endif

float evaluate_dense_f32_scalar(uint linear_index) {
    float lhs = uintBitsToFloat(input0_data.values[runtime_dense_input0_offset() + linear_index]);
    float rhs = uintBitsToFloat(input1_data.values[runtime_dense_input1_offset() + linear_index]);
    float result = apply_float(lhs, rhs, selected_mode);
#if ELTWISE_FUSED
    float fused_input = uintBitsToFloat(fused_input_data.values[dense_metadata.fused_input_offset + linear_index]);
    lhs = selected_fused_input_position == fused_input_lhs ? fused_input : result;
    rhs = selected_fused_input_position == fused_input_rhs ? fused_input : result;
    result = apply_fused_float(lhs, rhs);
#endif
    return result;
}

void evaluate_and_store_dense_f32_scalar(uint linear_index) {
#if ELTWISE_FUSED_CHAIN
    uint output_offset;
    output_data.values[runtime_dense_output_offset() + linear_index] = evaluate_element(linear_index, output_offset).x;
#else
    output_data.values[runtime_dense_output_offset() + linear_index] = floatBitsToUint(evaluate_dense_f32_scalar(linear_index));
#endif
}

void main() {
    const uint vector_width = ELTWISE_F32_VECTOR_WIDTH;
    uint first_element = gl_GlobalInvocationID.x * vector_width;
#if ELTWISE_F32_NO_TAIL
    ELTWISE_FLOAT_VECTOR lhs = load_f32_vector_0(runtime_dense_input0_offset() + first_element);
    ELTWISE_FLOAT_VECTOR rhs = load_f32_vector_1(runtime_dense_input1_offset() + first_element);
    ELTWISE_FLOAT_VECTOR result = apply_float_vector(lhs, rhs, selected_mode);
#if ELTWISE_FUSED
    ELTWISE_FLOAT_VECTOR fused_input = load_f32_vector_fused(dense_metadata.fused_input_offset + first_element);
    lhs = selected_fused_input_position == fused_input_lhs ? fused_input : result;
    rhs = selected_fused_input_position == fused_input_rhs ? fused_input : result;
    result = apply_fused_float_vector(lhs, rhs);
#elif ELTWISE_FUSED_CHAIN
    result = evaluate_fused_chain_float_vector(result, first_element);
#endif
    store_f32_vector(runtime_dense_output_offset() + first_element, result);
#else
    uint element_count = runtime_element_count();
    if (first_element >= element_count) {
        return;
    }

    uint remaining_elements = element_count - first_element;
    if (remaining_elements >= vector_width) {
        ELTWISE_FLOAT_VECTOR lhs = load_f32_vector_0(runtime_dense_input0_offset() + first_element);
        ELTWISE_FLOAT_VECTOR rhs = load_f32_vector_1(runtime_dense_input1_offset() + first_element);
        ELTWISE_FLOAT_VECTOR result = apply_float_vector(lhs, rhs, selected_mode);
#if ELTWISE_FUSED
        ELTWISE_FLOAT_VECTOR fused_input = load_f32_vector_fused(dense_metadata.fused_input_offset + first_element);
        lhs = selected_fused_input_position == fused_input_lhs ? fused_input : result;
        rhs = selected_fused_input_position == fused_input_rhs ? fused_input : result;
        result = apply_fused_float_vector(lhs, rhs);
#elif ELTWISE_FUSED_CHAIN
        result = evaluate_fused_chain_float_vector(result, first_element);
#endif
        store_f32_vector(runtime_dense_output_offset() + first_element, result);
        return;
    }
    evaluate_and_store_dense_f32_scalar(first_element);
    if (remaining_elements > 1) {
        evaluate_and_store_dense_f32_scalar(first_element + 1);
    }
#if ELTWISE_F32_VECTOR_WIDTH == 4
    if (remaining_elements > 2) {
        evaluate_and_store_dense_f32_scalar(first_element + 2);
    }
#endif
#endif
}

#undef ELTWISE_BOOL_VECTOR
#undef ELTWISE_UINT_VECTOR
#undef ELTWISE_FLOAT_VECTOR
