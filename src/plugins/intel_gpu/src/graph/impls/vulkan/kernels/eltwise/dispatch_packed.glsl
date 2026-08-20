// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if ELTWISE_PACKED_VECTOR_WIDTH == 2
#define ELTWISE_PACKED_UINT_VECTOR uvec2
#define ELTWISE_PACKED_INT_VECTOR ivec2
#define ELTWISE_PACKED_BOOL_VECTOR bvec2
#else
#define ELTWISE_PACKED_UINT_VECTOR uvec4
#define ELTWISE_PACKED_INT_VECTOR ivec4
#define ELTWISE_PACKED_BOOL_VECTOR bvec4
#endif

ELTWISE_PACKED_UINT_VECTOR unpack_packed_word(uint word) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_UINT_VECTOR(word & half_value_mask, word >> 16);
#else
    return ELTWISE_PACKED_UINT_VECTOR(word & byte_value_mask,
                                      (word >> 8) & byte_value_mask,
                                      (word >> 16) & byte_value_mask,
                                      word >> 24);
#endif
}

uint pack_packed_word(ELTWISE_PACKED_UINT_VECTOR value) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return (value.x & half_value_mask) | ((value.y & half_value_mask) << 16);
#else
    return (value.x & byte_value_mask) | ((value.y & byte_value_mask) << 8) |
           ((value.z & byte_value_mask) << 16) | ((value.w & byte_value_mask) << 24);
#endif
}

ELTWISE_PACKED_UINT_VECTOR sign_extend_packed_vector(ELTWISE_PACKED_UINT_VECTOR value) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_UINT_VECTOR(ELTWISE_PACKED_INT_VECTOR(value << 16) >> 16);
#else
    return ELTWISE_PACKED_UINT_VECTOR(ELTWISE_PACKED_INT_VECTOR(value << 24) >> 24);
#endif
}

ELTWISE_PACKED_UINT_VECTOR packed_boolean_bits(ELTWISE_PACKED_BOOL_VECTOR value) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_UINT_VECTOR(value.x ? 1u : 0u, value.y ? 1u : 0u);
#else
    return ELTWISE_PACKED_UINT_VECTOR(value.x ? 1u : 0u,
                                      value.y ? 1u : 0u,
                                      value.z ? 1u : 0u,
                                      value.w ? 1u : 0u);
#endif
}

ELTWISE_PACKED_BOOL_VECTOR packed_logical_and(ELTWISE_PACKED_BOOL_VECTOR lhs, ELTWISE_PACKED_BOOL_VECTOR rhs) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_BOOL_VECTOR(lhs.x && rhs.x, lhs.y && rhs.y);
#else
    return ELTWISE_PACKED_BOOL_VECTOR(lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z, lhs.w && rhs.w);
#endif
}

ELTWISE_PACKED_BOOL_VECTOR packed_logical_or(ELTWISE_PACKED_BOOL_VECTOR lhs, ELTWISE_PACKED_BOOL_VECTOR rhs) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_BOOL_VECTOR(lhs.x || rhs.x, lhs.y || rhs.y);
#else
    return ELTWISE_PACKED_BOOL_VECTOR(lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z, lhs.w || rhs.w);
#endif
}

ELTWISE_PACKED_BOOL_VECTOR packed_logical_xor(ELTWISE_PACKED_BOOL_VECTOR lhs, ELTWISE_PACKED_BOOL_VECTOR rhs) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_BOOL_VECTOR(lhs.x != rhs.x, lhs.y != rhs.y);
#else
    return ELTWISE_PACKED_BOOL_VECTOR(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w);
#endif
}

ELTWISE_PACKED_UINT_VECTOR select_packed_vector(ELTWISE_PACKED_UINT_VECTOR when_false,
                                                ELTWISE_PACKED_UINT_VECTOR when_true,
                                                ELTWISE_PACKED_BOOL_VECTOR condition) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
    return ELTWISE_PACKED_UINT_VECTOR(condition.x ? when_true.x : when_false.x,
                                      condition.y ? when_true.y : when_false.y);
#else
    return ELTWISE_PACKED_UINT_VECTOR(condition.x ? when_true.x : when_false.x,
                                      condition.y ? when_true.y : when_false.y,
                                      condition.z ? when_true.z : when_false.z,
                                      condition.w ? when_true.w : when_false.w);
#endif
}

uvec2 packed_scalar_bits(uint value, bool signed_type) {
    return uvec2(value, signed_type && int(value) < 0 ? 0xffffffffu : 0u);
}

ELTWISE_PACKED_UINT_VECTOR apply_integer_packed_scalar_fallback(ELTWISE_PACKED_UINT_VECTOR lhs,
                                                                ELTWISE_PACKED_UINT_VECTOR rhs,
                                                                bool signed_type,
                                                                bool fused) {
#if ELTWISE_PACKED_VECTOR_WIDTH == 2
#if ELTWISE_FUSED
    uvec2 result0 = fused ? apply_fused_integer(packed_scalar_bits(lhs.x, signed_type), packed_scalar_bits(rhs.x, signed_type), signed_type)
                          : apply_integer(packed_scalar_bits(lhs.x, signed_type), packed_scalar_bits(rhs.x, signed_type), selected_mode, signed_type);
    uvec2 result1 = fused ? apply_fused_integer(packed_scalar_bits(lhs.y, signed_type), packed_scalar_bits(rhs.y, signed_type), signed_type)
                          : apply_integer(packed_scalar_bits(lhs.y, signed_type), packed_scalar_bits(rhs.y, signed_type), selected_mode, signed_type);
#else
    uvec2 result0 = apply_integer(packed_scalar_bits(lhs.x, signed_type), packed_scalar_bits(rhs.x, signed_type), selected_mode, signed_type);
    uvec2 result1 = apply_integer(packed_scalar_bits(lhs.y, signed_type), packed_scalar_bits(rhs.y, signed_type), selected_mode, signed_type);
#endif
    return ELTWISE_PACKED_UINT_VECTOR(result0.x, result1.x);
#else
#if ELTWISE_FUSED
    uvec2 result0 = fused ? apply_fused_integer(packed_scalar_bits(lhs.x, signed_type), packed_scalar_bits(rhs.x, signed_type), signed_type)
                          : apply_integer(packed_scalar_bits(lhs.x, signed_type), packed_scalar_bits(rhs.x, signed_type), selected_mode, signed_type);
    uvec2 result1 = fused ? apply_fused_integer(packed_scalar_bits(lhs.y, signed_type), packed_scalar_bits(rhs.y, signed_type), signed_type)
                          : apply_integer(packed_scalar_bits(lhs.y, signed_type), packed_scalar_bits(rhs.y, signed_type), selected_mode, signed_type);
    uvec2 result2 = fused ? apply_fused_integer(packed_scalar_bits(lhs.z, signed_type), packed_scalar_bits(rhs.z, signed_type), signed_type)
                          : apply_integer(packed_scalar_bits(lhs.z, signed_type), packed_scalar_bits(rhs.z, signed_type), selected_mode, signed_type);
    uvec2 result3 = fused ? apply_fused_integer(packed_scalar_bits(lhs.w, signed_type), packed_scalar_bits(rhs.w, signed_type), signed_type)
                          : apply_integer(packed_scalar_bits(lhs.w, signed_type), packed_scalar_bits(rhs.w, signed_type), selected_mode, signed_type);
#else
    uvec2 result0 = apply_integer(packed_scalar_bits(lhs.x, signed_type), packed_scalar_bits(rhs.x, signed_type), selected_mode, signed_type);
    uvec2 result1 = apply_integer(packed_scalar_bits(lhs.y, signed_type), packed_scalar_bits(rhs.y, signed_type), selected_mode, signed_type);
    uvec2 result2 = apply_integer(packed_scalar_bits(lhs.z, signed_type), packed_scalar_bits(rhs.z, signed_type), selected_mode, signed_type);
    uvec2 result3 = apply_integer(packed_scalar_bits(lhs.w, signed_type), packed_scalar_bits(rhs.w, signed_type), selected_mode, signed_type);
#endif
    return ELTWISE_PACKED_UINT_VECTOR(result0.x, result1.x, result2.x, result3.x);
#endif
}

ELTWISE_PACKED_UINT_VECTOR apply_integer_packed(ELTWISE_PACKED_UINT_VECTOR lhs,
                                                ELTWISE_PACKED_UINT_VECTOR rhs,
                                                bool signed_type,
                                                bool fused) {
    uint mode = selected_mode;
#if ELTWISE_FUSED
    mode = fused ? selected_fused_mode : mode;
#endif
    switch (mode) {
    case mode_sum:
        return lhs + rhs;
    case mode_sub:
        return lhs - rhs;
    case mode_max: {
        ELTWISE_PACKED_BOOL_VECTOR less = signed_type ? lessThan(ELTWISE_PACKED_INT_VECTOR(lhs), ELTWISE_PACKED_INT_VECTOR(rhs)) : lessThan(lhs, rhs);
        return select_packed_vector(lhs, rhs, less);
    }
    case mode_prod:
        return lhs * rhs;
    case mode_min: {
        ELTWISE_PACKED_BOOL_VECTOR less = signed_type ? lessThan(ELTWISE_PACKED_INT_VECTOR(lhs), ELTWISE_PACKED_INT_VECTOR(rhs)) : lessThan(lhs, rhs);
        return select_packed_vector(rhs, lhs, less);
    }
    case mode_squared_diff: {
        ELTWISE_PACKED_UINT_VECTOR difference = lhs - rhs;
        return difference * difference;
    }
    case mode_eq:
        return packed_boolean_bits(equal(lhs, rhs));
    case mode_ne:
        return packed_boolean_bits(notEqual(lhs, rhs));
    case mode_lt:
        return packed_boolean_bits(signed_type ? lessThan(ELTWISE_PACKED_INT_VECTOR(lhs), ELTWISE_PACKED_INT_VECTOR(rhs)) : lessThan(lhs, rhs));
    case mode_le:
        return packed_boolean_bits(signed_type ? lessThanEqual(ELTWISE_PACKED_INT_VECTOR(lhs), ELTWISE_PACKED_INT_VECTOR(rhs)) : lessThanEqual(lhs, rhs));
    case mode_gt:
        return packed_boolean_bits(signed_type ? greaterThan(ELTWISE_PACKED_INT_VECTOR(lhs), ELTWISE_PACKED_INT_VECTOR(rhs)) : greaterThan(lhs, rhs));
    case mode_ge:
        return packed_boolean_bits(signed_type ? greaterThanEqual(ELTWISE_PACKED_INT_VECTOR(lhs), ELTWISE_PACKED_INT_VECTOR(rhs)) : greaterThanEqual(lhs, rhs));
    case mode_logic_and:
        return packed_boolean_bits(packed_logical_and(notEqual(lhs, ELTWISE_PACKED_UINT_VECTOR(0u)),
                                                      notEqual(rhs, ELTWISE_PACKED_UINT_VECTOR(0u))));
    case mode_logic_or:
        return packed_boolean_bits(packed_logical_or(notEqual(lhs, ELTWISE_PACKED_UINT_VECTOR(0u)),
                                                     notEqual(rhs, ELTWISE_PACKED_UINT_VECTOR(0u))));
    case mode_logic_xor:
        return packed_boolean_bits(packed_logical_xor(notEqual(lhs, ELTWISE_PACKED_UINT_VECTOR(0u)),
                                                      notEqual(rhs, ELTWISE_PACKED_UINT_VECTOR(0u))));
    case mode_bitwise_and:
        return lhs & rhs;
    case mode_bitwise_or:
        return lhs | rhs;
    case mode_bitwise_xor:
        return lhs ^ rhs;
    default:
        return apply_integer_packed_scalar_fallback(lhs, rhs, signed_type, fused);
    }
}

#if ELTWISE_PACKED_FLOAT16
vec2 apply_f16_packed(vec2 lhs, vec2 rhs, bool fused) {
    uint mode = selected_mode;
#if ELTWISE_FUSED
    mode = fused ? selected_fused_mode : mode;
#endif
    switch (mode) {
    case mode_sum:
#if ELTWISE_FUSED
        if (fused) {
            return lhs * vec2(uintBitsToFloat(runtime_fused_input0_coefficient())) +
                   rhs * vec2(uintBitsToFloat(runtime_fused_input1_coefficient()));
        }
#endif
        return lhs * vec2(uintBitsToFloat(runtime_input0_coefficient())) +
               rhs * vec2(uintBitsToFloat(runtime_input1_coefficient()));
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
    case mode_squared_diff: {
        vec2 difference = lhs - rhs;
        return difference * difference;
    }
    case mode_atan2:
        return atan(lhs, rhs);
    default:
#if ELTWISE_FUSED
        if (fused) {
            return vec2(apply_fused_float(lhs.x, rhs.x), apply_fused_float(lhs.y, rhs.y));
        }
#endif
        return vec2(apply_float(lhs.x, rhs.x, mode), apply_float(lhs.y, rhs.y, mode));
    }
}
#endif

void main() {
    const uint vector_width = ELTWISE_PACKED_VECTOR_WIDTH;
    uint packed_index = gl_GlobalInvocationID.x;
    uint packed_count = runtime_element_count() / vector_width;
    if (packed_index >= packed_count) {
        return;
    }

    uint input0_word = input0_data.values[dense_metadata.input0_offset / vector_width + packed_index];
    uint input1_word = input1_data.values[dense_metadata.input1_offset / vector_width + packed_index];
    uint output_word;
#if ELTWISE_PACKED_FLOAT16
    vec2 result = unpackHalf2x16(input0_word);
    result = apply_f16_packed(result, unpackHalf2x16(input1_word), false);
#if ELTWISE_FUSED
    uint fused_word = fused_input_data.values[dense_metadata.fused_input_offset / vector_width + packed_index];
    vec2 fused_value = unpackHalf2x16(fused_word);
    vec2 lhs = selected_fused_input_position == fused_input_lhs ? fused_value : result;
    vec2 rhs = selected_fused_input_position == fused_input_rhs ? fused_value : result;
    result = apply_f16_packed(lhs, rhs, true);
#endif
    output_word = packHalf2x16(result);
#else
    bool signed_type = is_signed_type(selected_input0_type);
    ELTWISE_PACKED_UINT_VECTOR lhs = unpack_packed_word(input0_word);
    ELTWISE_PACKED_UINT_VECTOR rhs = unpack_packed_word(input1_word);
    if (signed_type) {
        lhs = sign_extend_packed_vector(lhs);
        rhs = sign_extend_packed_vector(rhs);
    }
    ELTWISE_PACKED_UINT_VECTOR result = apply_integer_packed(lhs, rhs, signed_type, false);
#if ELTWISE_FUSED
    uint fused_word = fused_input_data.values[dense_metadata.fused_input_offset / vector_width + packed_index];
    ELTWISE_PACKED_UINT_VECTOR fused_value = unpack_packed_word(fused_word);
    if (signed_type) {
        fused_value = sign_extend_packed_vector(fused_value);
    }
    lhs = selected_fused_input_position == fused_input_lhs ? fused_value : result;
    rhs = selected_fused_input_position == fused_input_rhs ? fused_value : result;
    result = apply_integer_packed(lhs, rhs, signed_type, true);
#endif
    output_word = pack_packed_word(result);
#endif
    output_data.values[dense_metadata.output_offset / vector_width + packed_index] = output_word;
}

#undef ELTWISE_PACKED_BOOL_VECTOR
#undef ELTWISE_PACKED_INT_VECTOR
#undef ELTWISE_PACKED_UINT_VECTOR
