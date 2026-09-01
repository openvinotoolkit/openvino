// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

uint scalar_size(uint type) {
    if (type == type_i64) {
        return 8;
    }
    if (type == type_f32 || type == type_i32 || type == type_u32) {
        return 4;
    }
    if (type == type_f16 || type == type_i16 || type == type_u16) {
        return 2;
    }
    return 1;
}

uint load_u8_0(uint offset) {
#if ELTWISE_DENSE
    uint word = input0_data.values[offset / 4];
    return (word >> ((offset % 4) * 8)) & byte_value_mask;
#else
    return uint(input0_data.values[offset]);
#endif
}

#if !ELTWISE_UNARY
uint load_u8_1(uint offset) {
#if ELTWISE_DENSE
    uint word = input1_data.values[offset / 4];
    return (word >> ((offset % 4) * 8)) & byte_value_mask;
#else
    return uint(input1_data.values[offset]);
#endif
}
#endif

uint load_u16_0(uint offset) {
    return load_u8_0(offset) | (load_u8_0(offset + 1) << 8);
}

#if !ELTWISE_UNARY
uint load_u16_1(uint offset) {
    return load_u8_1(offset) | (load_u8_1(offset + 1) << 8);
}
#endif

uint load_u32_0(uint offset) {
#if ELTWISE_DENSE
    return input0_data.values[offset / 4];
#else
    return load_u16_0(offset) | (load_u16_0(offset + 2) << 16);
#endif
}

#if !ELTWISE_UNARY
uint load_u32_1(uint offset) {
#if ELTWISE_DENSE
    return input1_data.values[offset / 4];
#else
    return load_u16_1(offset) | (load_u16_1(offset + 2) << 16);
#endif
}
#endif

uvec2 load_u64_0(uint offset) {
    return uvec2(load_u32_0(offset), load_u32_0(offset + 4));
}

#if !ELTWISE_UNARY
uvec2 load_u64_1(uint offset) {
    return uvec2(load_u32_1(offset), load_u32_1(offset + 4));
}
#endif

#if ELTWISE_FUSED
uint load_u8_fused(uint offset) {
    uint word = fused_input_data.values[offset / 4];
    return (word >> ((offset % 4) * 8)) & byte_value_mask;
}

uint load_u16_fused(uint offset) {
    return load_u8_fused(offset) | (load_u8_fused(offset + 1) << 8);
}

uint load_u32_fused(uint offset) {
    return fused_input_data.values[offset / 4];
}

uvec2 load_u64_fused(uint offset) {
    return uvec2(load_u32_fused(offset), load_u32_fused(offset + 4));
}
#endif

#if ELTWISE_FUSED_CHAIN
uint load_u32_fused_chain(uint stage, uint offset) {
    uint word = offset / 4;
    switch (stage) {
    case 0:
        return fused_input0_data.values[word];
    case 1:
        return fused_input1_data.values[word];
    case 2:
        return fused_input2_data.values[word];
    case 3:
        return fused_input3_data.values[word];
    case 4:
        return fused_input4_data.values[word];
    case 5:
        return fused_input5_data.values[word];
    case 6:
        return fused_input6_data.values[word];
    case 7:
        return fused_input7_data.values[word];
    default:
        return 0;
    }
}

uint load_u8_fused_chain(uint stage, uint offset) {
    uint word = load_u32_fused_chain(stage, offset);
    return (word >> ((offset % 4) * 8)) & byte_value_mask;
}

uint load_u16_fused_chain(uint stage, uint offset) {
    return load_u8_fused_chain(stage, offset) | (load_u8_fused_chain(stage, offset + 1) << 8);
}

uvec2 load_u64_fused_chain(uint stage, uint offset) {
    return uvec2(load_u32_fused_chain(stage, offset), load_u32_fused_chain(stage, offset + 4));
}
#endif

void store_tail_u8(uint offset, uint value) {
#if ELTWISE_DENSE
    output_data.values[offset / 4] = value & byte_value_mask;
#else
    output_data.values[offset] = uint8_t(value & 0xff);
#endif
}

void store_tail_u16(uint offset, uint value) {
    store_tail_u8(offset, value);
    store_tail_u8(offset + 1, value >> 8);
}

uvec2 sign_extend(uint value, uint bits) {
    if (bits == 32) {
        return uvec2(value, uint(int(value) >> 31));
    }
    uint sign_bit = 1u << (bits - 1);
    uint mask = (1u << bits) - 1;
    uint extended = (value & sign_bit) != 0 ? value | ~mask : value & mask;
    return uvec2(extended, uint(int(extended) >> 31));
}

bool is_signed_type(uint type) {
    return type == type_i8 || type == type_i16 || type == type_i32 || type == type_i64;
}

bool is_float_type(uint type) {
    return type == type_f16 || type == type_f32;
}

uvec2 load_integer_0(uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_i64) {
        return load_u64_0(offset);
    }
    if (type == type_i32) {
        return sign_extend(load_u32_0(offset), 32);
    }
    if (type == type_u32) {
        return uvec2(load_u32_0(offset), 0);
    }
    if (type == type_i16) {
        return sign_extend(load_u16_0(offset), 16);
    }
    if (type == type_u16) {
        return uvec2(load_u16_0(offset), 0);
    }
    if (type == type_i8) {
        return sign_extend(load_u8_0(offset), 8);
    }
    return uvec2(load_u8_0(offset), 0);
}

#if !ELTWISE_UNARY
uvec2 load_integer_1(uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_i64) {
        return load_u64_1(offset);
    }
    if (type == type_i32) {
        return sign_extend(load_u32_1(offset), 32);
    }
    if (type == type_u32) {
        return uvec2(load_u32_1(offset), 0);
    }
    if (type == type_i16) {
        return sign_extend(load_u16_1(offset), 16);
    }
    if (type == type_u16) {
        return uvec2(load_u16_1(offset), 0);
    }
    if (type == type_i8) {
        return sign_extend(load_u8_1(offset), 8);
    }
    return uvec2(load_u8_1(offset), 0);
}
#endif

#if ELTWISE_FUSED
uvec2 load_integer_fused(uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_i64) {
        return load_u64_fused(offset);
    }
    if (type == type_i32) {
        return sign_extend(load_u32_fused(offset), 32);
    }
    if (type == type_u32) {
        return uvec2(load_u32_fused(offset), 0);
    }
    if (type == type_i16) {
        return sign_extend(load_u16_fused(offset), 16);
    }
    if (type == type_u16) {
        return uvec2(load_u16_fused(offset), 0);
    }
    if (type == type_i8) {
        return sign_extend(load_u8_fused(offset), 8);
    }
    return uvec2(load_u8_fused(offset), 0);
}
#endif

#if ELTWISE_FUSED_CHAIN
uvec2 load_integer_fused_chain(uint stage, uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_i64) {
        return load_u64_fused_chain(stage, offset);
    }
    if (type == type_i32) {
        return sign_extend(load_u32_fused_chain(stage, offset), 32);
    }
    if (type == type_u32) {
        return uvec2(load_u32_fused_chain(stage, offset), 0);
    }
    if (type == type_i16) {
        return sign_extend(load_u16_fused_chain(stage, offset), 16);
    }
    if (type == type_u16) {
        return uvec2(load_u16_fused_chain(stage, offset), 0);
    }
    if (type == type_i8) {
        return sign_extend(load_u8_fused_chain(stage, offset), 8);
    }
    return uvec2(load_u8_fused_chain(stage, offset), 0);
}
#endif

float signed_u64_to_float(uvec2 value) {
    bool negative = int(value.y) < 0;
    if (negative) {
        value = uvec2(~value.x + 1, ~value.y + (value.x == 0 ? 1 : 0));
    }
    float magnitude = float(value.y) * 4294967296.0 + float(value.x);
    return negative ? -magnitude : magnitude;
}

float unsigned_u64_to_float(uvec2 value) {
    return float(value.y) * 4294967296.0 + float(value.x);
}

#if ELTWISE_SCALAR_CONSTANT
uint scalar_constant_input_index() {
    return (selected_storage_flags & storage_scalar_constant_input0_flag) != 0 ? tensor_input0 : tensor_input1;
}

uvec2 scalar_constant_bits(uint type) {
    uint metadata_base = header_words + scalar_constant_input_index() * tensor_words;
    uvec2 value = uvec2(metadata.values[metadata_base], metadata.values[metadata_base + max_rank]);
    if (type == type_i64) {
        return value;
    }
    if (type == type_i32) {
        return sign_extend(value.x, 32);
    }
    if (type == type_i16) {
        return sign_extend(value.x, 16);
    }
    if (type == type_i8) {
        return sign_extend(value.x, 8);
    }
    return uvec2(value.x, 0);
}

float scalar_constant_float(uint type) {
    uvec2 value = scalar_constant_bits(type);
    if (type == type_f32) {
        return uintBitsToFloat(value.x);
    }
    if (type == type_f16) {
        return unpackHalf2x16(value.x).x;
    }
    return is_signed_type(type) ? signed_u64_to_float(value) : unsigned_u64_to_float(value);
}
#endif

float load_float_0(uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_f32) {
        return uintBitsToFloat(load_u32_0(offset));
    }
    if (type == type_f16) {
        return unpackHalf2x16(load_u16_0(offset)).x;
    }
    uvec2 value = load_integer_0(element_offset, type);
    return is_signed_type(type) ? signed_u64_to_float(value) : unsigned_u64_to_float(value);
}

#if !ELTWISE_UNARY
float load_float_1(uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_f32) {
        return uintBitsToFloat(load_u32_1(offset));
    }
    if (type == type_f16) {
        return unpackHalf2x16(load_u16_1(offset)).x;
    }
    uvec2 value = load_integer_1(element_offset, type);
    return is_signed_type(type) ? signed_u64_to_float(value) : unsigned_u64_to_float(value);
}
#endif

#if ELTWISE_FUSED
float load_float_fused(uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_f32) {
        return uintBitsToFloat(load_u32_fused(offset));
    }
    if (type == type_f16) {
        return unpackHalf2x16(load_u16_fused(offset)).x;
    }
    uvec2 value = load_integer_fused(element_offset, type);
    return is_signed_type(type) ? signed_u64_to_float(value) : unsigned_u64_to_float(value);
}
#endif

#if ELTWISE_FUSED_CHAIN
float load_float_fused_chain(uint stage, uint element_offset, uint type) {
    uint offset = element_offset * scalar_size(type);
    if (type == type_f32) {
        return uintBitsToFloat(load_u32_fused_chain(stage, offset));
    }
    if (type == type_f16) {
        return unpackHalf2x16(load_u16_fused_chain(stage, offset)).x;
    }
    uvec2 value = load_integer_fused_chain(stage, element_offset, type);
    return is_signed_type(type) ? signed_u64_to_float(value) : unsigned_u64_to_float(value);
}
#endif
