// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

uvec2 add_u64(uvec2 lhs, uvec2 rhs) {
    uint low = lhs.x + rhs.x;
    return uvec2(low, lhs.y + rhs.y + (low < lhs.x ? 1 : 0));
}

uvec2 negate_u64(uvec2 value) {
    uint low = ~value.x + 1;
    return uvec2(low, ~value.y + (low == 0 ? 1 : 0));
}

uvec2 sub_u64(uvec2 lhs, uvec2 rhs) {
    return add_u64(lhs, negate_u64(rhs));
}

uvec2 mul_u64(uvec2 lhs, uvec2 rhs) {
    uint high_product;
    uint low_product;
    umulExtended(lhs.x, rhs.x, high_product, low_product);
    return uvec2(low_product, high_product + lhs.x * rhs.y + lhs.y * rhs.x);
}

bool equal_u64(uvec2 lhs, uvec2 rhs) {
    return all(equal(lhs, rhs));
}

bool zero_u64(uvec2 value) {
    return value.x == 0 && value.y == 0;
}

bool less_unsigned_u64(uvec2 lhs, uvec2 rhs) {
    return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
}

bool less_signed_u64(uvec2 lhs, uvec2 rhs) {
    int lhs_high = int(lhs.y);
    int rhs_high = int(rhs.y);
    return lhs_high < rhs_high || (lhs_high == rhs_high && lhs.x < rhs.x);
}

uvec2 shift_left_u64(uvec2 value, uint amount) {
    if (amount >= 64) {
        return uvec2(0);
    }
    if (amount >= 32) {
        return uvec2(0, value.x << (amount - 32));
    }
    if (amount == 0) {
        return value;
    }
    return uvec2(value.x << amount, (value.y << amount) | (value.x >> (32 - amount)));
}

uvec2 shift_right_unsigned_u64(uvec2 value, uint amount) {
    if (amount >= 64) {
        return uvec2(0);
    }
    if (amount >= 32) {
        return uvec2(value.y >> (amount - 32), 0);
    }
    if (amount == 0) {
        return value;
    }
    return uvec2((value.x >> amount) | (value.y << (32 - amount)), value.y >> amount);
}

uvec2 shift_right_signed_u64(uvec2 value, uint amount) {
    if (amount >= 64) {
        return int(value.y) < 0 ? uvec2(0xffffffff) : uvec2(0);
    }
    if (amount >= 32) {
        return uvec2(uint(int(value.y) >> int(amount - 32)), uint(int(value.y) >> 31));
    }
    if (amount == 0) {
        return value;
    }
    return uvec2((value.x >> amount) | (value.y << (32 - amount)), uint(int(value.y) >> int(amount)));
}

uint bit_u64(uvec2 value, uint index) {
    return index < 32 ? (value.x >> index) & 1 : (value.y >> (index - 32)) & 1;
}

uvec2 set_bit_u64(uvec2 value, uint index) {
    if (index < 32) {
        value.x |= 1u << index;
    } else {
        value.y |= 1u << (index - 32);
    }
    return value;
}

void divide_unsigned_u64(uvec2 numerator, uvec2 denominator, out uvec2 quotient, out uvec2 remainder) {
    quotient = uvec2(0);
    remainder = uvec2(0);
    if (zero_u64(denominator)) {
        return;
    }
    for (int index = 63; index >= 0; --index) {
        remainder = shift_left_u64(remainder, 1);
        remainder.x |= bit_u64(numerator, uint(index));
        if (!less_unsigned_u64(remainder, denominator)) {
            remainder = sub_u64(remainder, denominator);
            quotient = set_bit_u64(quotient, uint(index));
        }
    }
}

void divide_integer(uvec2 lhs, uvec2 rhs, bool signed_type, bool floor_division, out uvec2 quotient, out uvec2 remainder) {
    bool lhs_negative = signed_type && int(lhs.y) < 0;
    bool rhs_negative = signed_type && int(rhs.y) < 0;
    uvec2 lhs_abs = lhs_negative ? negate_u64(lhs) : lhs;
    uvec2 rhs_abs = rhs_negative ? negate_u64(rhs) : rhs;
    divide_unsigned_u64(lhs_abs, rhs_abs, quotient, remainder);
    if (lhs_negative != rhs_negative) {
        quotient = negate_u64(quotient);
    }
    if (lhs_negative) {
        remainder = negate_u64(remainder);
    }
    if (floor_division && !zero_u64(remainder) && lhs_negative != rhs_negative) {
        quotient = sub_u64(quotient, uvec2(1, 0));
        remainder = add_u64(remainder, rhs);
    }
}

uvec2 pow_u64(uvec2 base, uvec2 exponent) {
    if (int(exponent.y) < 0) {
        return uvec2(0);
    }
    uvec2 result = uvec2(1, 0);
    while (!zero_u64(exponent)) {
        if ((exponent.x & 1) != 0) {
            result = mul_u64(result, base);
        }
        base = mul_u64(base, base);
        exponent = shift_right_unsigned_u64(exponent, 1);
    }
    return result;
}
