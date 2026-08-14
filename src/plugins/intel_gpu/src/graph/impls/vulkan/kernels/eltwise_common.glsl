#if !ELTWISE_DENSE
#extension GL_EXT_shader_8bit_storage : require
#endif
#extension GL_GOOGLE_include_directive : require

#ifndef ELTWISE_DENSE_PUSH_CONSTANTS
#define ELTWISE_DENSE_PUSH_CONSTANTS 0
#endif

#ifndef ELTWISE_F32_VECTOR_WIDTH
#define ELTWISE_F32_VECTOR_WIDTH 0
#endif

#ifndef ELTWISE_F32_NO_TAIL
#define ELTWISE_F32_NO_TAIL 0
#endif

#ifndef ELTWISE_PACKED_VECTOR_WIDTH
#define ELTWISE_PACKED_VECTOR_WIDTH 0
#endif

#ifndef ELTWISE_PACKED_FLOAT16
#define ELTWISE_PACKED_FLOAT16 0
#endif

#ifndef ELTWISE_FAST_BROADCAST
#define ELTWISE_FAST_BROADCAST 0
#endif

#ifndef ELTWISE_FUSED_CHAIN
#define ELTWISE_FUSED_CHAIN 0
#endif

#ifndef ELTWISE_FUSED_BROADCAST
#define ELTWISE_FUSED_BROADCAST 0
#endif

#ifndef ELTWISE_FIXED_FUSED_CHAIN_LENGTH
#define ELTWISE_FIXED_FUSED_CHAIN_LENGTH 0
#endif

#if ELTWISE_RESTRICT_OUTPUT && !ELTWISE_DENSE
#error ELTWISE_RESTRICT_OUTPUT requires the single-output dense ABI
#endif

#if ELTWISE_RESTRICT_OUTPUT
#define ELTWISE_OUTPUT_ALIAS_QUALIFIER restrict
#else
#define ELTWISE_OUTPUT_ALIAS_QUALIFIER
#endif

#if ELTWISE_FUSED_CHAIN
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint values[];
} input1_data;

layout(set = 0, binding = 2) readonly buffer FusedInput0 {
    uint values[];
} fused_input0_data;
layout(set = 0, binding = 3) readonly buffer FusedInput1 {
    uint values[];
} fused_input1_data;
layout(set = 0, binding = 4) readonly buffer FusedInput2 {
    uint values[];
} fused_input2_data;
layout(set = 0, binding = 5) readonly buffer FusedInput3 {
    uint values[];
} fused_input3_data;
layout(set = 0, binding = 6) readonly buffer FusedInput4 {
    uint values[];
} fused_input4_data;
layout(set = 0, binding = 7) readonly buffer FusedInput5 {
    uint values[];
} fused_input5_data;
layout(set = 0, binding = 8) readonly buffer FusedInput6 {
    uint values[];
} fused_input6_data;
layout(set = 0, binding = 9) readonly buffer FusedInput7 {
    uint values[];
} fused_input7_data;

layout(set = 0, binding = 10) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint values[];
} output_data;
#elif ELTWISE_FUSED
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint values[];
} input1_data;

layout(set = 0, binding = 2) readonly buffer FusedInput {
    uint values[];
} fused_input_data;

layout(set = 0, binding = 3) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint values[];
} output_data;
#elif ELTWISE_UNARY
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint8_t values[];
} input0_data;

layout(set = 0, binding = 1) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint8_t values[];
} output_data;
#elif ELTWISE_SCALAR_CONSTANT
layout(set = 0, binding = 0) readonly buffer TensorInput {
    uint8_t values[];
} tensor_input_data;
#define input0_data tensor_input_data
#define input1_data tensor_input_data

layout(set = 0, binding = 1) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint8_t values[];
} output_data;
#elif ELTWISE_DENSE
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint values[];
} input1_data;

layout(set = 0, binding = 2) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint values[];
} output_data;
#else
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint8_t values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint8_t values[];
} input1_data;

layout(set = 0, binding = 2) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint8_t values[];
} output_data;
#endif

#if ELTWISE_DENSE_PUSH_CONSTANTS
layout(push_constant) uniform DenseMetadata {
    uint element_count;
    uint input0_offset;
    uint input1_offset;
    uint output_offset;
    uint python_division;
    uint infinity_detection;
    uint input0_coefficient;
    uint input1_coefficient;
#if ELTWISE_FUSED
    uint fused_input_offset;
    uint fused_python_division;
    uint fused_input0_coefficient;
    uint fused_input1_coefficient;
#endif
} dense_metadata;
#else
#if ELTWISE_FUSED_CHAIN
layout(set = 0, binding = 11) readonly buffer Metadata {
#elif ELTWISE_FUSED
layout(set = 0, binding = 4) readonly buffer Metadata {
#elif ELTWISE_UNARY || ELTWISE_SCALAR_CONSTANT
layout(set = 0, binding = 2) readonly buffer Metadata {
#else
layout(set = 0, binding = 3) readonly buffer Metadata {
#endif
    uint values[];
} metadata;
#endif

#if ELTWISE_FUSED || ELTWISE_DENSE
#define packed_output_data output_data
#elif ELTWISE_UNARY || ELTWISE_SCALAR_CONSTANT
layout(set = 0, binding = 3) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer PackedOutput {
    uint values[];
} packed_output_data;
#else
layout(set = 0, binding = 4) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer PackedOutput {
    uint values[];
} packed_output_data;
#endif

#undef ELTWISE_OUTPUT_ALIAS_QUALIFIER

#define ELTWISE_SHADER_MODE(name, code) const uint mode_##name = code;
#define ELTWISE_SHADER_SCALAR_TYPE(name, code) const uint type_##name = code;
#define ELTWISE_SHADER_METADATA_FIELD(name, code) const uint metadata_##name = code;
#define ELTWISE_SHADER_TENSOR_INDEX(name, code) const uint tensor_##name = code;
#define ELTWISE_SHADER_STORAGE_FLAG(name, code) const uint storage_##name##_flag = code;
#define ELTWISE_SHADER_INFINITY_FLAG(name, code) const uint infinity_##name##_flag = code;
#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
#define ELTWISE_SHADER_FUSED_INPUT_POSITION(name, code) const uint fused_input_##name = code;
#endif
#define ELTWISE_SHADER_LIMIT(name, code) const uint name = code;
#define ELTWISE_SHADER_SPECIALIZATION_ID(name, code) const uint specialization_##name##_id = code;
#include "../eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_SPECIALIZATION_ID
#undef ELTWISE_SHADER_LIMIT
#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
#undef ELTWISE_SHADER_FUSED_INPUT_POSITION
#endif
#undef ELTWISE_SHADER_INFINITY_FLAG
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
#include "../eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_METADATA_FIELD
#define ELTWISE_SHADER_FUSED_METADATA_FIELD(name, code) const uint metadata_fused_##name = fused_metadata_base + fused_metadata_##name;
#include "../eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_METADATA_FIELD
const uint fused_chain_metadata_base = fused_metadata_base + max_fused_chain_length * fused_metadata_count;
#define ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD(name, code) const uint fused_chain_metadata_##name = code;
#include "../eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD
#define ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD(name, code) const uint metadata_fused_chain_##name = fused_chain_metadata_base + fused_chain_metadata_##name;
#include "../eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD
#if ELTWISE_FUSED_CHAIN
const uint fast_divisor_metadata_base = fused_chain_metadata_base + fused_chain_metadata_count;
#else
const uint fast_divisor_metadata_base = fused_metadata_base + fused_metadata_count;
#endif
const uint fused_broadcast_metadata_base = fused_chain_metadata_base + fused_chain_metadata_count + max_rank;
const uint byte_value_mask = 0xff;
const uint half_value_mask = 0xffff;

uint runtime_element_count() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.element_count;
#else
    return metadata.values[metadata_element_count];
#endif
}

uint runtime_python_division() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.python_division;
#else
    return metadata.values[metadata_python_division];
#endif
}

uint runtime_infinity_detection() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.infinity_detection;
#else
    return metadata.values[metadata_infinity_detection];
#endif
}

uint runtime_input0_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input0_coefficient;
#else
    return metadata.values[metadata_input0_coefficient];
#endif
}

uint runtime_input1_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input1_coefficient;
#else
    return metadata.values[metadata_input1_coefficient];
#endif
}

uint runtime_dense_input0_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input0_offset;
#else
    return metadata.values[header_words + tensor_input0 * tensor_words + max_rank * 2];
#endif
}

uint runtime_dense_input1_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input1_offset;
#else
    return metadata.values[header_words + tensor_input1 * tensor_words + max_rank * 2];
#endif
}

uint runtime_dense_output_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.output_offset;
#else
    return metadata.values[header_words + tensor_output * tensor_words + max_rank * 2];
#endif
}

#if ELTWISE_FUSED
uint runtime_fused_input_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_input_offset;
#else
    return metadata.values[metadata_fused_input_offset];
#endif
}

uint runtime_fused_python_division() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_python_division;
#else
    return metadata.values[metadata_fused_python_division];
#endif
}

uint runtime_fused_input0_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_input0_coefficient;
#else
    return metadata.values[metadata_fused_input0_coefficient];
#endif
}

uint runtime_fused_input1_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_input1_coefficient;
#else
    return metadata.values[metadata_fused_input1_coefficient];
#endif
}
#endif

#if ELTWISE_FUSED_CHAIN
uint fused_chain_metadata(uint stage, uint field) {
    return metadata.values[fused_metadata_base + stage * fused_metadata_count + field];
}

uint runtime_fused_chain_length() {
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH > 0
    return uint(ELTWISE_FIXED_FUSED_CHAIN_LENGTH);
#else
    return metadata.values[metadata_fused_chain_chain_length];
#endif
}
#endif

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

#if !ELTWISE_DENSE || ELTWISE_FUSED_BROADCAST
uint storage_element_offset_from_base(uint base, uint coordinates[max_rank], uint rank) {
    uint offset = metadata.values[base + max_rank * 2];
    for (uint axis = 0; axis < rank; ++axis) {
        uint dimension = metadata.values[base + axis];
        uint pitch = metadata.values[base + max_rank + axis];
        offset += (dimension == 1 ? 0 : coordinates[axis]) * pitch;
    }
    return offset;
}

#if !ELTWISE_DENSE
uint storage_element_offset(uint tensor_index, uint coordinates[max_rank], uint rank) {
    return storage_element_offset_from_base(header_words + tensor_index * tensor_words, coordinates, rank);
}
#endif

#if ELTWISE_FAST_BROADCAST || ELTWISE_FUSED_BROADCAST
uint fast_divide_u32(uint numerator, uint divisor, uint reciprocal) {
    if (divisor == 1) {
        return numerator;
    }
    uint product_high;
    uint product_low;
    umulExtended(numerator, reciprocal, product_high, product_low);
    uint remainder = numerator - product_high * divisor;
    return product_high + (remainder >= divisor ? 1u : 0u);
}

#endif

#if ELTWISE_FAST_BROADCAST
void fast_broadcast_offsets(uint linear_index, out uint input0_offset, out uint input1_offset, out uint output_offset) {
    uint input0_base = header_words + tensor_input0 * tensor_words;
    uint input1_base = header_words + tensor_input1 * tensor_words;
    uint output_base = header_words + tensor_output * tensor_words;
    input0_offset = metadata.values[input0_base + max_rank * 2];
    input1_offset = metadata.values[input1_base + max_rank * 2];
    output_offset = metadata.values[output_base + max_rank * 2];

    uint remainder = linear_index;
    for (int axis = int(selected_broadcast_rank) - 1; axis >= 0; --axis) {
        uint unsigned_axis = uint(axis);
        uint dimension = metadata.values[output_base + unsigned_axis];
        uint reciprocal = metadata.values[fast_divisor_metadata_base + unsigned_axis];
        uint quotient = fast_divide_u32(remainder, dimension, reciprocal);
        uint coordinate = remainder - quotient * dimension;
        uint axis_bit = 1u << unsigned_axis;
        if ((selected_broadcast_input0_axes & axis_bit) != 0) {
            input0_offset += coordinate * metadata.values[input0_base + max_rank + unsigned_axis];
        }
        if ((selected_broadcast_input1_axes & axis_bit) != 0) {
            input1_offset += coordinate * metadata.values[input1_base + max_rank + unsigned_axis];
        }
        if ((selected_broadcast_output_axes & axis_bit) != 0) {
            output_offset += coordinate * metadata.values[output_base + max_rank + unsigned_axis];
        }
        remainder = quotient;
    }
}
#endif
#endif

#if ELTWISE_FUSED_BROADCAST
uint fused_broadcast_input_offset(uint linear_index) {
    if ((selected_storage_flags & storage_fused_scalar_input_flag) != 0) {
        return metadata.values[fused_broadcast_metadata_base + max_rank * 2];
    }
    uint rank = metadata.values[metadata_rank];
    uint output_base = header_words + tensor_output * tensor_words;
    uint coordinates[max_rank];
    uint remainder = linear_index;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        uint unsigned_axis = uint(axis);
        uint dimension = metadata.values[output_base + unsigned_axis];
        uint reciprocal = metadata.values[fast_divisor_metadata_base + unsigned_axis];
        uint quotient = fast_divide_u32(remainder, dimension, reciprocal);
        coordinates[unsigned_axis] = remainder - quotient * dimension;
        remainder = quotient;
    }
    return storage_element_offset_from_base(fused_broadcast_metadata_base, coordinates, rank);
}
#endif

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

#if ELTWISE_FUSED || ELTWISE_FUSED_CHAIN
uvec2 evaluate_base_element(uint linear_index, out uint output_offset) {
#else
uvec2 evaluate_element(uint linear_index, out uint output_offset) {
#endif
    uint mode = selected_mode;
    uint input0_type = selected_input0_type;
#if !ELTWISE_UNARY
    uint input1_type = selected_input1_type;
#endif
    uint output_type = selected_output_type;
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

#if ELTWISE_FUSED
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
#elif ELTWISE_FUSED_CHAIN
uvec2 evaluate_fused_chain_stage(uvec2 result,
                                 uint stage,
                                 uint linear_index,
                                 uint output_type,
                                 uint mode,
                                 uint input_type,
                                 uint input_position) {
    uint input_offset = fused_chain_metadata(stage, fused_metadata_input_offset) + linear_index;
    if (is_float_type(output_type)) {
        float original = output_type == type_f32 ? uintBitsToFloat(result.x) : unpackHalf2x16(result.x).x;
        float external_value = load_float_fused_chain(stage, input_offset, input_type);
        float lhs = input_position == fused_input_lhs ? external_value : original;
        float rhs = input_position == fused_input_rhs ? external_value : original;
        return float_output_bits(apply_fused_float_runtime(lhs,
                                                           rhs,
                                                           mode,
                                                           fused_chain_metadata(stage, fused_metadata_input0_coefficient),
                                                           fused_chain_metadata(stage, fused_metadata_input1_coefficient)),
                                 output_type);
    }

    uvec2 external_value = load_integer_fused_chain(stage, input_offset, input_type);
    uvec2 lhs = input_position == fused_input_lhs ? external_value : result;
    uvec2 rhs = input_position == fused_input_rhs ? external_value : result;
    return apply_fused_integer_runtime(lhs,
                                       rhs,
                                       is_signed_type(output_type),
                                       mode,
                                       fused_chain_metadata(stage, fused_metadata_python_division) != 0);
}

uvec2 evaluate_element(uint linear_index, out uint output_offset) {
    uvec2 result = evaluate_base_element(linear_index, output_offset);
    uint output_type = selected_output_type;
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH > 0
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 1
    result = evaluate_fused_chain_stage(result,
                                        0,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_0,
                                        selected_fused_chain_input_type_0,
                                        selected_fused_chain_input_position_0);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 2
    result = evaluate_fused_chain_stage(result,
                                        1,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_1,
                                        selected_fused_chain_input_type_1,
                                        selected_fused_chain_input_position_1);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 3
    result = evaluate_fused_chain_stage(result,
                                        2,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_2,
                                        selected_fused_chain_input_type_2,
                                        selected_fused_chain_input_position_2);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 4
    result = evaluate_fused_chain_stage(result,
                                        3,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_3,
                                        selected_fused_chain_input_type_3,
                                        selected_fused_chain_input_position_3);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 5
    result = evaluate_fused_chain_stage(result,
                                        4,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_4,
                                        selected_fused_chain_input_type_4,
                                        selected_fused_chain_input_position_4);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 6
    result = evaluate_fused_chain_stage(result,
                                        5,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_5,
                                        selected_fused_chain_input_type_5,
                                        selected_fused_chain_input_position_5);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 7
    result = evaluate_fused_chain_stage(result,
                                        6,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_6,
                                        selected_fused_chain_input_type_6,
                                        selected_fused_chain_input_position_6);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 8
    result = evaluate_fused_chain_stage(result,
                                        7,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_7,
                                        selected_fused_chain_input_type_7,
                                        selected_fused_chain_input_position_7);
#endif
#else
    uint chain_length = runtime_fused_chain_length();
    for (uint stage = 0; stage < chain_length; ++stage) {
        result = evaluate_fused_chain_stage(result,
                                            stage,
                                            linear_index,
                                            output_type,
                                            fused_chain_metadata(stage, fused_metadata_mode),
                                            fused_chain_metadata(stage, fused_metadata_input_type),
                                            fused_chain_metadata(stage, fused_metadata_input_position));
    }
#endif
    return result;
}
#endif

void store_element(uint output_offset, uint output_size, uvec2 value) {
    uint output_byte_offset = output_offset * output_size;
    if (output_size == 1) {
        store_tail_u8(output_byte_offset, value.x);
    } else if (output_size == 2) {
        store_tail_u16(output_byte_offset, value.x);
    } else {
        uint output_word = output_byte_offset / 4;
        packed_output_data.values[output_word] = value.x;
        if (output_size == 8) {
            packed_output_data.values[output_word + 1] = value.y;
        }
    }
}

#if ELTWISE_PACKED_VECTOR_WIDTH == 2 || ELTWISE_PACKED_VECTOR_WIDTH == 4
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
#elif ELTWISE_F32_VECTOR_WIDTH == 2 || ELTWISE_F32_VECTOR_WIDTH == 4
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
#else
void main() {
    uint invocation_index = gl_GlobalInvocationID.x;
    uint element_count = runtime_element_count();
    uint output_type = selected_output_type;
    uint output_size = scalar_size(output_type);
    bool pack_output = output_size < 4 && (selected_storage_flags & storage_packed_output_flag) != 0;
#if ELTWISE_DENSE || ELTWISE_BROADCAST_VECTOR || ELTWISE_SCALAR_CONSTANT
    uint elements_per_invocation = selected_elements_per_invocation;
#else
    uint elements_per_invocation = pack_output ? 4 / output_size : 1;
#endif
    uint first_element = invocation_index * elements_per_invocation;
    if (first_element >= element_count) {
        return;
    }

    uint first_output_offset;
    uvec2 first_value = evaluate_element(first_element, first_output_offset);
    if (!pack_output) {
        store_element(first_output_offset, output_size, first_value);
#if ELTWISE_DENSE || ELTWISE_BROADCAST_VECTOR || ELTWISE_SCALAR_CONSTANT
        for (uint vector_index = 1; vector_index < elements_per_invocation; ++vector_index) {
            uint element_index = first_element + vector_index;
            if (element_index >= element_count) {
                break;
            }
            uint output_offset;
            uvec2 value = evaluate_element(element_index, output_offset);
            store_element(output_offset, output_size, value);
        }
#endif
        return;
    }

    uint component_mask = output_size == 2 ? half_value_mask : byte_value_mask;
    uint packed_value = first_value.x & component_mask;
    uint processed_elements = 1;
    for (uint packed_index = 1; packed_index < elements_per_invocation; ++packed_index) {
        uint element_index = first_element + packed_index;
        if (element_index >= element_count) {
            break;
        }
        uint output_offset;
        uvec2 value = evaluate_element(element_index, output_offset);
        packed_value |= (value.x & component_mask) << (packed_index * output_size * 8);
        processed_elements += 1;
    }

    uint output_byte_offset = first_output_offset * output_size;
    if (processed_elements == elements_per_invocation) {
        packed_output_data.values[output_byte_offset / 4] = packed_value;
    } else if (output_size == 2) {
        store_tail_u16(output_byte_offset, packed_value);
    } else {
        for (uint byte_index = 0; byte_index < processed_elements; ++byte_index) {
            store_tail_u8(output_byte_offset + byte_index, packed_value >> (byte_index * 8));
        }
    }
}
#endif
