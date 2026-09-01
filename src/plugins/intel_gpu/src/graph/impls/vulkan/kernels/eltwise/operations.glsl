// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

bool is_boolean_mode(uint mode) {
    return (mode >= mode_eq && mode <= mode_logic_xor) || (mode >= mode_is_finite && mode <= mode_is_nan);
}

float quiet_nan() {
    return 0.0 / 0.0;
}

float truncate_toward_zero(float value) {
    return value < 0.0 ? ceil(value) : floor(value);
}

float apply_mod(float lhs, float rhs) {
    if (rhs == 0.0 || isinf(lhs)) {
        return quiet_nan();
    }
    if (lhs == 0.0 || isinf(rhs)) {
        return lhs;
    }
    if (abs(lhs) == abs(rhs)) {
        return lhs * 0.0;
    }
    return lhs - truncate_toward_zero(lhs / rhs) * rhs;
}

float apply_pow(float lhs, float rhs) {
    if (rhs == 0.0) {
        return 1.0;
    }
    if (lhs >= 0.0) {
        return pow(lhs, rhs);
    }

    float integer_exponent = truncate_toward_zero(rhs);
    if (integer_exponent != rhs) {
        return quiet_nan();
    }

    float magnitude = pow(-lhs, rhs);
    float exponent_pairs = truncate_toward_zero(integer_exponent * 0.5);
    bool odd_exponent = integer_exponent != exponent_pairs * 2.0;
    return odd_exponent ? -magnitude : magnitude;
}

float apply_float(float lhs, float rhs, uint mode) {
    switch (mode) {
    case mode_sum:
        return lhs * uintBitsToFloat(runtime_input0_coefficient()) + rhs * uintBitsToFloat(runtime_input1_coefficient());
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
        return apply_pow(lhs, rhs);
    case mode_squared_diff: {
        float difference = lhs - rhs;
        return difference * difference;
    }
    case mode_mod:
        return apply_mod(lhs, rhs);
    case mode_floor_mod:
        return rhs == 0.0 || lhs == rhs ? 0.0 : lhs - floor(lhs / rhs) * rhs;
    case mode_atan2:
        return atan(lhs, rhs);
    default:
        return 0.0;
    }
}

bool apply_float_boolean(float lhs, float rhs, uint mode) {
    switch (mode) {
    case mode_eq:
        return lhs == rhs;
    case mode_ne:
        return lhs != rhs;
    case mode_lt:
        return lhs < rhs;
    case mode_le:
        return lhs <= rhs;
    case mode_gt:
        return lhs > rhs;
    case mode_ge:
        return lhs >= rhs;
    case mode_logic_and:
        return lhs != 0.0 && rhs != 0.0;
    case mode_logic_or:
        return lhs != 0.0 || rhs != 0.0;
    case mode_logic_xor:
        return (lhs != 0.0) != (rhs != 0.0);
    case mode_is_finite:
        return !isnan(lhs) && !isinf(lhs);
    case mode_is_inf: {
        uint detect_mask = runtime_infinity_detection();
        return isinf(lhs) && ((lhs < 0.0 && (detect_mask & infinity_negative_flag) != 0) ||
                              (lhs > 0.0 && (detect_mask & infinity_positive_flag) != 0));
    }
    case mode_is_nan:
        return isnan(lhs);
    default:
        return false;
    }
}

#if ELTWISE_UNARY
bool apply_float_unary_boolean(float value, uint mode) {
    switch (mode) {
    case mode_is_finite:
        return !isnan(value) && !isinf(value);
    case mode_is_inf: {
        uint detect_mask = runtime_infinity_detection();
        return isinf(value) && ((value < 0.0 && (detect_mask & infinity_negative_flag) != 0) ||
                                (value > 0.0 && (detect_mask & infinity_positive_flag) != 0));
    }
    case mode_is_nan:
        return isnan(value);
    default:
        return false;
    }
}
#endif

uvec2 apply_integer(uvec2 lhs, uvec2 rhs, uint mode, bool signed_type) {
    switch (mode) {
    case mode_sum:
        return add_u64(lhs, rhs);
    case mode_sub:
        return sub_u64(lhs, rhs);
    case mode_max:
        return (signed_type ? less_signed_u64(lhs, rhs) : less_unsigned_u64(lhs, rhs)) ? rhs : lhs;
    case mode_prod:
        return mul_u64(lhs, rhs);
    case mode_div: {
        uvec2 quotient;
        uvec2 remainder;
        divide_integer(lhs, rhs, signed_type, runtime_python_division() != 0, quotient, remainder);
        return quotient;
    }
    case mode_min:
        return (signed_type ? less_signed_u64(lhs, rhs) : less_unsigned_u64(lhs, rhs)) ? lhs : rhs;
    case mode_pow:
        return pow_u64(lhs, rhs);
    case mode_squared_diff: {
        uvec2 difference = sub_u64(lhs, rhs);
        return mul_u64(difference, difference);
    }
    case mode_mod: {
        uvec2 quotient;
        uvec2 remainder;
        divide_integer(lhs, rhs, signed_type, false, quotient, remainder);
        return remainder;
    }
    case mode_floor_mod: {
        uvec2 quotient;
        uvec2 remainder;
        divide_integer(lhs, rhs, signed_type, true, quotient, remainder);
        return remainder;
    }
    case mode_right_shift:
        return signed_type ? shift_right_signed_u64(lhs, rhs.x) : shift_right_unsigned_u64(lhs, rhs.x);
    case mode_left_shift:
        return shift_left_u64(lhs, rhs.x);
    case mode_bitwise_and:
        return lhs & rhs;
    case mode_bitwise_or:
        return lhs | rhs;
    case mode_bitwise_xor:
        return lhs ^ rhs;
    default:
        return uvec2(0);
    }
}

#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
float apply_fused_float_runtime(float lhs, float rhs, uint mode, uint input0_coefficient, uint input1_coefficient) {
    if (mode == mode_sum) {
        return lhs * uintBitsToFloat(input0_coefficient) + rhs * uintBitsToFloat(input1_coefficient);
    }
    return apply_float(lhs, rhs, mode);
}

uvec2 apply_fused_integer_runtime(uvec2 lhs, uvec2 rhs, bool signed_type, uint mode, bool python_division) {
    switch (mode) {
    case mode_sum:
        return add_u64(lhs, rhs);
    case mode_sub:
        return sub_u64(lhs, rhs);
    case mode_prod:
        return mul_u64(lhs, rhs);
    case mode_div: {
        uvec2 quotient;
        uvec2 remainder;
        divide_integer(lhs,
                       rhs,
                       signed_type,
                       python_division,
                       quotient,
                       remainder);
        return quotient;
    }
    default:
        return uvec2(0);
    }
}
#endif

#if ELTWISE_FUSED
float apply_fused_float(float lhs, float rhs) {
    return apply_fused_float_runtime(lhs,
                                     rhs,
                                     selected_fused_mode,
                                     runtime_fused_input0_coefficient(),
                                     runtime_fused_input1_coefficient());
}

uvec2 apply_fused_integer(uvec2 lhs, uvec2 rhs, bool signed_type) {
    return apply_fused_integer_runtime(lhs, rhs, signed_type, selected_fused_mode, runtime_fused_python_division() != 0);
}
#endif

bool apply_integer_boolean(uvec2 lhs, uvec2 rhs, uint mode, bool signed_type) {
    switch (mode) {
    case mode_eq:
        return equal_u64(lhs, rhs);
    case mode_ne:
        return !equal_u64(lhs, rhs);
    case mode_lt:
        return signed_type ? less_signed_u64(lhs, rhs) : less_unsigned_u64(lhs, rhs);
    case mode_le:
        return equal_u64(lhs, rhs) || (signed_type ? less_signed_u64(lhs, rhs) : less_unsigned_u64(lhs, rhs));
    case mode_gt:
        return !equal_u64(lhs, rhs) && !(signed_type ? less_signed_u64(lhs, rhs) : less_unsigned_u64(lhs, rhs));
    case mode_ge:
        return !(signed_type ? less_signed_u64(lhs, rhs) : less_unsigned_u64(lhs, rhs));
    case mode_logic_and:
        return !zero_u64(lhs) && !zero_u64(rhs);
    case mode_logic_or:
        return !zero_u64(lhs) || !zero_u64(rhs);
    case mode_logic_xor:
        return zero_u64(lhs) != zero_u64(rhs);
    case mode_is_finite:
        return true;
    case mode_is_inf:
    case mode_is_nan:
        return false;
    default:
        return false;
    }
}
