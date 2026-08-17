// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "activation_inst.h"
#include "data_inst.h"
#include "eltwise_broadcast_f32_eq_spirv.hpp"
#include "eltwise_broadcast_fast_f32_eq_spirv.hpp"
#include "eltwise_broadcast_fast_spirv.hpp"
#include "eltwise_broadcast_fast_vector_spirv.hpp"
#include "eltwise_broadcast_vector_spirv.hpp"
#include "eltwise_dense_f32_div_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_div_push_constants_spirv.hpp"
#include "eltwise_dense_f32_sum_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_sum_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec2_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec2_no_tail_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec2_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec2_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec4_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec4_no_tail_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec4_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec4_push_constants_spirv.hpp"
#include "eltwise_dense_i64_div_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_i64_div_push_constants_spirv.hpp"
#include "eltwise_dense_i64_sum_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_i64_sum_push_constants_spirv.hpp"
#include "eltwise_dense_packed_16bit_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_packed_16bit_push_constants_spirv.hpp"
#include "eltwise_dense_packed_8bit_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_packed_8bit_push_constants_spirv.hpp"
#include "eltwise_dense_packed_f16_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_packed_f16_push_constants_spirv.hpp"
#include "eltwise_dense_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_push_constants_spirv.hpp"
#include "eltwise_dense_restrict_spirv.hpp"
#include "eltwise_dense_spirv.hpp"
#include "eltwise_fused_broadcast_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_push_constants_spirv.hpp"
#include "eltwise_fused_dense_chain_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_spirv.hpp"
#include "eltwise_fused_dense_packed_16bit_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_packed_16bit_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_8bit_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_packed_8bit_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_f16_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_packed_f16_push_constants_spirv.hpp"
#include "eltwise_fused_dense_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_push_constants_spirv.hpp"
#include "eltwise_fused_dense_restrict_spirv.hpp"
#include "eltwise_fused_dense_spirv.hpp"
#include "eltwise_fused_post_op_spirv.hpp"
#include "eltwise_scalar_constant_spirv.hpp"
#include "eltwise_shader_abi.hpp"
#include "eltwise_spirv.hpp"
#include "eltwise_unary_spirv.hpp"
#include "common_utils/kernels_cache.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "quantize_inst.h"
#include "registry/implementation_map.hpp"
#include "reorder_inst.h"
#include "vulkan/vulkan_engine.hpp"
#include "vulkan/vulkan_memory.hpp"

namespace cldnn {
namespace vulkan {
namespace {

namespace shader_abi = eltwise_shader_abi;

constexpr uint32_t max_rank = 8;
constexpr uint32_t header_words = shader_abi::index(shader_abi::metadata_field::count);
constexpr uint32_t tensor_words = max_rank * 2 + 1;
constexpr uint32_t tensor_count = shader_abi::index(shader_abi::tensor_index::count);
constexpr uint32_t fused_metadata_base = header_words + tensor_count * tensor_words;
constexpr uint32_t min_multi_stage_fused_chain_length = 2;
constexpr uint32_t max_fused_chain_length = shader_abi::value(shader_abi::limit::max_fused_chain_length);
constexpr uint32_t fused_metadata_words = shader_abi::index(shader_abi::fused_metadata_field::count);
constexpr uint32_t regular_fast_divisor_metadata_base = fused_metadata_base + fused_metadata_words;
constexpr uint32_t regular_metadata_words = regular_fast_divisor_metadata_base + max_rank;
constexpr uint32_t fused_chain_metadata_base = fused_metadata_base + max_fused_chain_length * fused_metadata_words;
constexpr uint32_t fused_chain_fast_divisor_metadata_base =
    fused_chain_metadata_base + shader_abi::index(shader_abi::fused_chain_metadata_field::count);
constexpr uint32_t fused_chain_metadata_words = fused_chain_fast_divisor_metadata_base + max_rank;
constexpr uint32_t fused_broadcast_metadata_base = fused_chain_metadata_words;
constexpr uint32_t fused_broadcast_metadata_words = fused_broadcast_metadata_base + tensor_words;
constexpr uint32_t post_op_metadata_base = fused_broadcast_metadata_words;
constexpr uint32_t post_op_metadata_words =
    post_op_metadata_base + shader_abi::index(shader_abi::post_op_metadata_field::count);
constexpr uint32_t metadata_words = post_op_metadata_words;
constexpr uint32_t dense_push_constant_words = shader_abi::index(shader_abi::dense_metadata_field::count);
constexpr uint32_t fused_dense_push_constant_words = dense_push_constant_words + shader_abi::index(shader_abi::fused_dense_metadata_field::count);
constexpr uint32_t dense_push_constant_bytes = dense_push_constant_words * sizeof(uint32_t);
constexpr uint32_t fused_dense_push_constant_bytes = fused_dense_push_constant_words * sizeof(uint32_t);
constexpr uint32_t portable_max_local_work_group_size = 128;
constexpr uint32_t broadcast_vector_width = 4;
constexpr uint32_t broadcast_vector_min_elements = 256;
constexpr uint32_t maximum_scalar_batch_width = 8;
constexpr uint32_t balanced_scalar_batch_width = 4;
constexpr uint32_t division_scalar_batch_width = 2;
constexpr uint32_t scalar_batch_width = 1;
constexpr uint32_t scalar_elements_per_subgroup_budget = portable_max_local_work_group_size;
constexpr size_t local_size_tuning_samples_per_candidate = 3;
constexpr size_t local_size_stabilization_inferences = 20;

enum class kernel_kind : uint8_t {
    dense,
    broadcast_scalar,
    broadcast_vector,
    unary,
    scalar_constant,
    fused_dense,
    dense_push_constants,
    fused_dense_push_constants,
    dense_f32_vec2_push_constants,
    dense_f32_vec4_push_constants,
    fused_dense_f32_vec2_push_constants,
    fused_dense_f32_vec4_push_constants,
    dense_f32_vec2_no_tail_push_constants,
    dense_f32_vec4_no_tail_push_constants,
    fused_dense_f32_vec2_no_tail_push_constants,
    fused_dense_f32_vec4_no_tail_push_constants,
    dense_packed_8bit_push_constants,
    dense_packed_16bit_push_constants,
    fused_dense_packed_8bit_push_constants,
    fused_dense_packed_16bit_push_constants,
    dense_packed_f16_push_constants,
    fused_dense_packed_f16_push_constants,
    broadcast_fast_scalar,
    broadcast_fast_vector,
    dense_f32_sum_push_constants,
    dense_f32_div_push_constants,
    dense_i64_sum_push_constants,
    dense_i64_div_push_constants,
    broadcast_f32_eq,
    broadcast_fast_f32_eq,
    fused_dense_chain,
    fused_dense_chain_f32_vec4_no_tail,
    fused_broadcast,
    fused_post_op,
};

enum class dense_vector_width : uint32_t {
    scalar = 1,
    vec2 = 2,
    vec4 = 4,
};

enum class packed_dense_width : uint32_t {
    none = 0,
    two_16bit_elements = 2,
    four_8bit_elements = 4,
};

constexpr uint32_t value(dense_vector_width width) {
    return static_cast<uint32_t>(width);
}

bool is_f32_vector_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_f32_vec2_push_constants,
                   kernel_kind::dense_f32_vec4_push_constants,
                   kernel_kind::fused_dense_f32_vec2_push_constants,
                   kernel_kind::fused_dense_f32_vec4_push_constants,
                   kernel_kind::dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_chain_f32_vec4_no_tail});
}

bool is_no_tail_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_chain_f32_vec4_no_tail});
}

bool is_packed_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_packed_8bit_push_constants,
                   kernel_kind::dense_packed_16bit_push_constants,
                   kernel_kind::fused_dense_packed_8bit_push_constants,
                   kernel_kind::fused_dense_packed_16bit_push_constants,
                   kernel_kind::dense_packed_f16_push_constants,
                   kernel_kind::fused_dense_packed_f16_push_constants});
}

bool uses_dense_push_constants(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_push_constants,
                   kernel_kind::fused_dense_push_constants,
                   kernel_kind::dense_f32_sum_push_constants,
                   kernel_kind::dense_f32_div_push_constants,
                   kernel_kind::dense_i64_sum_push_constants,
                   kernel_kind::dense_i64_div_push_constants}) ||
           (is_f32_vector_kernel(kind) && kind != kernel_kind::fused_dense_chain_f32_vec4_no_tail) || is_packed_dense_kernel(kind);
}

bool is_broadcast_vector_kernel(kernel_kind kind) {
    return one_of(kind, {kernel_kind::broadcast_vector, kernel_kind::broadcast_fast_vector});
}

bool is_fast_broadcast_kernel(kernel_kind kind) {
    return one_of(kind, {kernel_kind::broadcast_fast_scalar, kernel_kind::broadcast_fast_vector, kernel_kind::broadcast_fast_f32_eq});
}

bool is_plain_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense,
                   kernel_kind::dense_push_constants,
                   kernel_kind::dense_f32_sum_push_constants,
                   kernel_kind::dense_f32_div_push_constants,
                   kernel_kind::dense_i64_sum_push_constants,
                   kernel_kind::dense_i64_div_push_constants,
                   kernel_kind::dense_f32_vec2_push_constants,
                   kernel_kind::dense_f32_vec4_push_constants,
                   kernel_kind::dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::dense_packed_8bit_push_constants,
                   kernel_kind::dense_packed_16bit_push_constants,
                   kernel_kind::dense_packed_f16_push_constants});
}

bool is_single_fused_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::fused_dense,
                   kernel_kind::fused_dense_push_constants,
                   kernel_kind::fused_dense_f32_vec2_push_constants,
                   kernel_kind::fused_dense_f32_vec4_push_constants,
                   kernel_kind::fused_dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_packed_8bit_push_constants,
                   kernel_kind::fused_dense_packed_16bit_push_constants,
                   kernel_kind::fused_dense_packed_f16_push_constants});
}

bool is_fused_broadcast_kernel(kernel_kind kind) {
    return kind == kernel_kind::fused_broadcast;
}

bool is_fused_post_op_kernel(kernel_kind kind) {
    return kind == kernel_kind::fused_post_op;
}

bool is_single_fused_kernel(kernel_kind kind) {
    return is_single_fused_dense_kernel(kind) || is_fused_broadcast_kernel(kind);
}

bool is_fused_chain_kernel(kernel_kind kind) {
    return one_of(kind, {kernel_kind::fused_dense_chain, kernel_kind::fused_dense_chain_f32_vec4_no_tail});
}

bool is_fused_dense_kernel(kernel_kind kind) {
    return is_single_fused_dense_kernel(kind) || is_fused_chain_kernel(kind);
}

bool is_fused_kernel(kernel_kind kind) {
    return is_fused_dense_kernel(kind) || is_fused_broadcast_kernel(kind);
}

bool supports_restricted_output(kernel_kind kind) {
    return is_plain_dense_kernel(kind) || is_fused_dense_kernel(kind);
}

dense_vector_width get_dense_vector_width(kernel_kind kind) {
    if (one_of(kind,
               {kernel_kind::dense_f32_vec4_push_constants,
                kernel_kind::fused_dense_f32_vec4_push_constants,
                kernel_kind::dense_f32_vec4_no_tail_push_constants,
                kernel_kind::fused_dense_f32_vec4_no_tail_push_constants,
                kernel_kind::fused_dense_chain_f32_vec4_no_tail})) {
        return dense_vector_width::vec4;
    }
    if (one_of(kind,
               {kernel_kind::dense_f32_vec2_push_constants,
                kernel_kind::fused_dense_f32_vec2_push_constants,
                kernel_kind::dense_f32_vec2_no_tail_push_constants,
                kernel_kind::fused_dense_f32_vec2_no_tail_push_constants})) {
        return dense_vector_width::vec2;
    }
    return dense_vector_width::scalar;
}

packed_dense_width get_packed_dense_width(kernel_kind kind) {
    if (one_of(kind, {kernel_kind::dense_packed_8bit_push_constants, kernel_kind::fused_dense_packed_8bit_push_constants})) {
        return packed_dense_width::four_8bit_elements;
    }
    if (one_of(kind,
               {kernel_kind::dense_packed_16bit_push_constants,
                kernel_kind::fused_dense_packed_16bit_push_constants,
                kernel_kind::dense_packed_f16_push_constants,
                kernel_kind::fused_dense_packed_f16_push_constants})) {
        return packed_dense_width::two_16bit_elements;
    }
    return packed_dense_width::none;
}

constexpr uint32_t value(packed_dense_width width) {
    return static_cast<uint32_t>(width);
}

struct scalar_constant {
    shader_abi::tensor_index input_index = shader_abi::tensor_index::input1;
    std::array<uint32_t, 2> bits{};
};

bool is_supported_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::sum,        eltwise_mode::sub,         eltwise_mode::max,          eltwise_mode::prod,       eltwise_mode::div,
                         eltwise_mode::min,        eltwise_mode::pow,         eltwise_mode::squared_diff, eltwise_mode::mod,        eltwise_mode::eq,
                         eltwise_mode::ne,         eltwise_mode::lt,          eltwise_mode::le,           eltwise_mode::gt,         eltwise_mode::ge,
                         eltwise_mode::logic_and,  eltwise_mode::logic_or,    eltwise_mode::logic_xor,    eltwise_mode::floor_mod,  eltwise_mode::is_finite,
                         eltwise_mode::is_inf,     eltwise_mode::is_nan,      eltwise_mode::right_shift,  eltwise_mode::left_shift, eltwise_mode::bitwise_and,
                         eltwise_mode::bitwise_or, eltwise_mode::bitwise_xor, eltwise_mode::atan2});
}

bool is_unary_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::is_finite, eltwise_mode::is_inf, eltwise_mode::is_nan});
}

bool is_fused_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::sum, eltwise_mode::prod, eltwise_mode::sub, eltwise_mode::div});
}

bool is_bitwise_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::right_shift, eltwise_mode::left_shift, eltwise_mode::bitwise_and, eltwise_mode::bitwise_or, eltwise_mode::bitwise_xor});
}

bool is_supported_data_type(data_types type) {
    return one_of(type,
                  {data_types::f16,
                   data_types::f32,
                   data_types::i8,
                   data_types::u8,
                   data_types::i16,
                   data_types::u16,
                   data_types::i32,
                   data_types::u32,
                   data_types::i64,
                   data_types::boolean});
}

bool is_integer_data_type(data_types type) {
    return one_of(type,
                  {data_types::i8, data_types::u8, data_types::i16, data_types::u16, data_types::i32, data_types::u32, data_types::i64, data_types::boolean});
}

shader_abi::mode shader_mode_code(eltwise_mode mode) {
    switch (mode) {
    case eltwise_mode::sum:
        return shader_abi::mode::sum;
    case eltwise_mode::sub:
        return shader_abi::mode::sub;
    case eltwise_mode::max:
        return shader_abi::mode::max;
    case eltwise_mode::prod:
        return shader_abi::mode::prod;
    case eltwise_mode::div:
        return shader_abi::mode::div;
    case eltwise_mode::min:
        return shader_abi::mode::min;
    case eltwise_mode::pow:
        return shader_abi::mode::pow;
    case eltwise_mode::squared_diff:
        return shader_abi::mode::squared_diff;
    case eltwise_mode::mod:
        return shader_abi::mode::mod;
    case eltwise_mode::eq:
        return shader_abi::mode::eq;
    case eltwise_mode::ne:
        return shader_abi::mode::ne;
    case eltwise_mode::lt:
        return shader_abi::mode::lt;
    case eltwise_mode::le:
        return shader_abi::mode::le;
    case eltwise_mode::gt:
        return shader_abi::mode::gt;
    case eltwise_mode::ge:
        return shader_abi::mode::ge;
    case eltwise_mode::logic_and:
        return shader_abi::mode::logic_and;
    case eltwise_mode::logic_or:
        return shader_abi::mode::logic_or;
    case eltwise_mode::logic_xor:
        return shader_abi::mode::logic_xor;
    case eltwise_mode::floor_mod:
        return shader_abi::mode::floor_mod;
    case eltwise_mode::is_finite:
        return shader_abi::mode::is_finite;
    case eltwise_mode::is_inf:
        return shader_abi::mode::is_inf;
    case eltwise_mode::is_nan:
        return shader_abi::mode::is_nan;
    case eltwise_mode::right_shift:
        return shader_abi::mode::right_shift;
    case eltwise_mode::left_shift:
        return shader_abi::mode::left_shift;
    case eltwise_mode::bitwise_and:
        return shader_abi::mode::bitwise_and;
    case eltwise_mode::bitwise_or:
        return shader_abi::mode::bitwise_or;
    case eltwise_mode::bitwise_xor:
        return shader_abi::mode::bitwise_xor;
    case eltwise_mode::atan2:
        return shader_abi::mode::atan2;
    default:
        OPENVINO_THROW("[GPU][Vulkan] Unsupported Eltwise shader mode");
    }
}

shader_abi::scalar_type scalar_type_code(data_types type) {
    switch (type) {
    case data_types::f16:
        return shader_abi::scalar_type::f16;
    case data_types::f32:
        return shader_abi::scalar_type::f32;
    case data_types::i8:
        return shader_abi::scalar_type::i8;
    case data_types::u8:
        return shader_abi::scalar_type::u8;
    case data_types::i16:
        return shader_abi::scalar_type::i16;
    case data_types::u16:
        return shader_abi::scalar_type::u16;
    case data_types::i32:
        return shader_abi::scalar_type::i32;
    case data_types::u32:
        return shader_abi::scalar_type::u32;
    case data_types::i64:
        return shader_abi::scalar_type::i64;
    case data_types::boolean:
        return shader_abi::scalar_type::boolean;
    default:
        OPENVINO_THROW("[GPU][Vulkan] Unsupported Eltwise scalar type ", ov::element::Type(type).get_type_name());
    }
}

uint32_t float_bits(float value) {
    uint32_t result = 0;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool is_supported_format(format::type fmt) {
    return one_of(fmt, {format::any, format::bfyx, format::yxfb, format::bfzyx, format::bfwzyx, format::bfuwzyx, format::bfvuwzyx});
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] Eltwise ", description, " exceeds the 32-bit shader metadata range");
    return static_cast<uint32_t>(value);
}

uint32_t select_local_work_group_size(uint32_t element_count, uint64_t device_max_work_group_size) {
    const auto limit = static_cast<uint32_t>(std::min<uint64_t>(portable_max_local_work_group_size, device_max_work_group_size));
    OPENVINO_ASSERT(limit > 0, "[GPU][Vulkan] Device reports a zero maximum work-group size");

    uint32_t local_size = 1;
    while (local_size < element_count && local_size <= limit / 2) {
        local_size *= 2;
    }
    return local_size;
}

struct local_size_tuning_state {
    std::mutex mutex;
    std::atomic<uint32_t> cached_selected{0};
    uint32_t invocation_count = 0;
    uint32_t fallback = 1;
    uint32_t selected = 1;
    std::vector<uint32_t> candidates;
    std::vector<std::vector<uint64_t>> samples;
    size_t fallback_observations = 0;
    bool prewarmed = false;
    bool decided = false;
};

struct local_size_tuning_action {
    uint32_t local_size = 1;
    size_t candidate_index = 0;
    bool prewarm = false;
    bool measure = false;
};

std::vector<uint32_t> make_local_size_candidates(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility) {
    const auto fallback = select_local_work_group_size(invocation_count, info.max_work_group_size);
    std::vector<uint32_t> candidates{fallback};
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0) {
        return candidates;
    }

    const auto limit = static_cast<uint32_t>(std::min<uint64_t>(info.max_work_group_size, std::numeric_limits<uint32_t>::max()));
    const auto subgroup_size = info.supported_simd_sizes.front();
    const auto add_candidate = [&](uint32_t candidate) {
        if (candidate >= subgroup_size && candidate % subgroup_size == 0 && candidate <= limit && candidate <= invocation_count &&
            (!require_exact_divisibility || invocation_count % candidate == 0)) {
            candidates.push_back(candidate);
        }
    };
    if (subgroup_size <= fallback / 2) {
        const auto midpoint = (fallback / 2 / subgroup_size) * subgroup_size;
        add_candidate(midpoint);
    }
    if (fallback <= limit / 2) {
        const auto expanded = ((fallback * 2 + subgroup_size - 1) / subgroup_size) * subgroup_size;
        add_candidate(expanded);
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

uint64_t median_local_size_sample(std::vector<uint64_t> samples) {
    const auto middle = samples.begin() + samples.size() / 2;
    std::nth_element(samples.begin(), middle, samples.end());
    return *middle;
}

bool local_size_diagnostics_enabled() {
    const auto* value = std::getenv("OV_GPU_VULKAN_LOCAL_SIZE_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void initialize_local_size_tuning(local_size_tuning_state& state, uint32_t invocation_count, const device_info& info, bool require_exact_divisibility) {
    state.invocation_count = invocation_count;
    state.fallback = select_local_work_group_size(invocation_count, info.max_work_group_size);
    state.selected = state.fallback;
    state.candidates = make_local_size_candidates(invocation_count, info, require_exact_divisibility);
    state.samples.assign(state.candidates.size(), {});
    state.fallback_observations = 0;
    state.prewarmed = false;
    state.decided = state.candidates.size() == 1;
    state.cached_selected.store(state.decided ? state.fallback : 0, std::memory_order_release);
}

void decide_local_size(local_size_tuning_state& state) {
    const auto fallback_it = std::find(state.candidates.begin(), state.candidates.end(), state.fallback);
    OPENVINO_ASSERT(fallback_it != state.candidates.end(), "[GPU][Vulkan] Eltwise local-size candidates lost the portable fallback");
    const auto fallback_index = static_cast<size_t>(std::distance(state.candidates.begin(), fallback_it));
    auto best_index = fallback_index;
    auto best_median = median_local_size_sample(state.samples[fallback_index]);
    for (size_t index = 0; index < state.candidates.size(); ++index) {
        const auto candidate_median = median_local_size_sample(state.samples[index]);
        if (candidate_median < best_median) {
            best_index = index;
            best_median = candidate_median;
        }
    }

    if (best_index != fallback_index) {
        const auto best_max = *std::max_element(state.samples[best_index].begin(), state.samples[best_index].end());
        const auto fallback_min = *std::min_element(state.samples[fallback_index].begin(), state.samples[fallback_index].end());
        if (best_max >= fallback_min) {
            best_index = fallback_index;
        }
    }
    state.selected = state.candidates[best_index];
    state.decided = true;
    state.cached_selected.store(state.selected, std::memory_order_release);

    if (local_size_diagnostics_enabled()) {
        std::clog << "[GPU][Vulkan][EltwiseLocalSize] invocations=" << state.invocation_count << " fallback=" << state.fallback
                  << " selected=" << state.selected;
        for (size_t index = 0; index < state.candidates.size(); ++index) {
            std::clog << " candidate_" << state.candidates[index] << "_median_ns=" << median_local_size_sample(state.samples[index]);
        }
        std::clog << std::endl;
    }
}

local_size_tuning_action next_local_size_action(local_size_tuning_state& state,
                                                uint32_t invocation_count,
                                                const device_info& info,
                                                bool require_exact_divisibility) {
    if (state.invocation_count != invocation_count || state.candidates.empty()) {
        initialize_local_size_tuning(state, invocation_count, info, require_exact_divisibility);
    }
    if (state.decided) {
        return {state.selected, 0, false, false};
    }
    if (state.fallback_observations < local_size_stabilization_inferences) {
        ++state.fallback_observations;
        return {state.fallback, 0, false, false};
    }
    if (!state.prewarmed) {
        return {state.fallback, 0, true, false};
    }

    size_t total_samples = 0;
    for (const auto& candidate_samples : state.samples) {
        total_samples += candidate_samples.size();
    }
    const auto sample_round = total_samples / state.candidates.size();
    if (sample_round == local_size_tuning_samples_per_candidate) {
        decide_local_size(state);
        return {state.selected, 0, false, false};
    }
    const auto position = total_samples % state.candidates.size();
    const auto candidate_index = sample_round % 2 == 0 ? position : state.candidates.size() - position - 1;
    return {state.candidates[candidate_index], candidate_index, false, true};
}

bool has_dense_storage(const layout& tensor_layout) {
    return !static_cast<bool>(tensor_layout.data_padding) &&
           tensor_layout.bytes_count() == tensor_layout.count() * data_type_traits::size_of(tensor_layout.data_type);
}

std::optional<scalar_constant> get_scalar_constant(const program_node& node) {
    for (uint32_t input_index = shader_abi::index(shader_abi::tensor_index::input0); input_index <= shader_abi::index(shader_abi::tensor_index::input1);
         ++input_index) {
        const auto& dependency = node.get_dependency(input_index);
        const auto& input_layout = dependency.get_output_layout(false);
        if (!dependency.is_type<data>() || !dependency.is_constant() || input_layout.is_dynamic() || input_layout.count() != 1 ||
            input_layout.get_linear_offset() != 0 || !has_dense_storage(input_layout)) {
            continue;
        }

        scalar_constant result;
        result.input_index = static_cast<shader_abi::tensor_index>(input_index);
        const auto value_size = data_type_traits::size_of(input_layout.data_type);
        OPENVINO_ASSERT(value_size <= sizeof(result.bits), "[GPU][Vulkan] Eltwise scalar constant exceeds the supported storage size");
        mem_lock<uint8_t, mem_lock_type::read> constant_data{dependency.as<data>().get_attached_memory_ptr(), node.get_program().get_stream()};
        std::memcpy(result.bits.data(), constant_data.data(), value_size);
        return result;
    }
    return std::nullopt;
}

bool can_use_scalar_linear_storage(const layout& tensor_layout, const layout& output_layout) {
    return has_dense_storage(tensor_layout) && has_dense_storage(output_layout) && tensor_layout.count() == output_layout.count();
}

bool can_use_linear_storage(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    return has_dense_storage(input0_layout) && has_dense_storage(input1_layout) && has_dense_storage(output_layout) && input0_layout.identical(output_layout) &&
           input1_layout.identical(output_layout);
}

uint32_t output_elements_per_invocation(const layout& output_layout) {
    const auto scalar_size = data_type_traits::size_of(output_layout.data_type);
    if (scalar_size >= sizeof(uint32_t) || !has_dense_storage(output_layout)) {
        return 1;
    }
    OPENVINO_ASSERT(sizeof(uint32_t) % scalar_size == 0, "[GPU][Vulkan] Eltwise output scalar size cannot be packed into a 32-bit storage word");
    return checked_u32(sizeof(uint32_t) / scalar_size, "packed output width");
}

bool can_use_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    return data_type_traits::size_of(output_layout.data_type) >= sizeof(uint32_t) && can_use_linear_storage(input0_layout, input1_layout, output_layout);
}

packed_dense_width packed_width_for_type(data_types type) {
    if (one_of(type, {data_types::i8, data_types::u8})) {
        return packed_dense_width::four_8bit_elements;
    }
    if (one_of(type, {data_types::f16, data_types::i16, data_types::u16})) {
        return packed_dense_width::two_16bit_elements;
    }
    return packed_dense_width::none;
}

bool is_aligned_for_packed_width(const layout& tensor_layout, packed_dense_width width) {
    return tensor_layout.get_linear_offset() % value(width) == 0;
}

bool can_use_packed_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout, const layout* fused_input_layout) {
    const auto width = packed_width_for_type(output_layout.data_type);
    if (width == packed_dense_width::none || input0_layout.data_type != output_layout.data_type || input1_layout.data_type != output_layout.data_type ||
        !can_use_linear_storage(input0_layout, input1_layout, output_layout) || output_layout.count() % value(width) != 0 ||
        !is_aligned_for_packed_width(input0_layout, width) || !is_aligned_for_packed_width(input1_layout, width) ||
        !is_aligned_for_packed_width(output_layout, width)) {
        return false;
    }
    return fused_input_layout == nullptr || (fused_input_layout->data_type == output_layout.data_type && fused_input_layout->identical(output_layout) &&
                                             has_dense_storage(*fused_input_layout) && is_aligned_for_packed_width(*fused_input_layout, width));
}

packed_dense_width select_packed_dense_width(const layout& input0_layout,
                                             const layout& input1_layout,
                                             const layout& output_layout,
                                             const layout* fused_input_layout,
                                             const device_info& info) {
    if (!can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, fused_input_layout) || info.supported_simd_sizes.empty() ||
        info.supported_simd_sizes.front() == 0) {
        return packed_dense_width::none;
    }
    const auto width = packed_width_for_type(output_layout.data_type);
    const auto logical_elements_per_subgroup = static_cast<uint64_t>(info.supported_simd_sizes.front()) * value(width);
    if (logical_elements_per_subgroup > portable_max_local_work_group_size) {
        return packed_dense_width::none;
    }
    const auto invocation_count = output_layout.count() / value(width);
    const auto useful_subgroup_width = std::min<uint64_t>(std::max<uint64_t>(info.max_work_group_size, 1), info.supported_simd_sizes.front());
    return invocation_count >= useful_subgroup_width ? width : packed_dense_width::none;
}

kernel_kind select_packed_dense_kernel_kind(packed_dense_width width, data_types type, bool fused) {
    if (width == packed_dense_width::four_8bit_elements) {
        return fused ? kernel_kind::fused_dense_packed_8bit_push_constants : kernel_kind::dense_packed_8bit_push_constants;
    }
    OPENVINO_ASSERT(width == packed_dense_width::two_16bit_elements, "[GPU][Vulkan] Invalid packed dense width selected for Eltwise");
    if (type == data_types::f16) {
        return fused ? kernel_kind::fused_dense_packed_f16_push_constants : kernel_kind::dense_packed_f16_push_constants;
    }
    return fused ? kernel_kind::fused_dense_packed_16bit_push_constants : kernel_kind::dense_packed_16bit_push_constants;
}

bool can_use_fused_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& fused_input_layout, const layout& output_layout) {
    if (!can_use_linear_storage(input0_layout, input1_layout, output_layout) || !has_dense_storage(fused_input_layout) ||
        !fused_input_layout.identical(output_layout)) {
        return false;
    }
    const auto packed_width = output_elements_per_invocation(output_layout);
    return packed_width == 1 || (output_layout.count() % packed_width == 0 && output_layout.get_linear_offset() % packed_width == 0);
}

bool is_aligned_for_vector_width(const layout& tensor_layout, dense_vector_width width) {
    return tensor_layout.get_linear_offset() % value(width) == 0;
}

bool can_use_f32_dense_vector_width(const layout& input0_layout,
                                    const layout& input1_layout,
                                    const layout& output_layout,
                                    const layout* fused_input_layout,
                                    dense_vector_width width) {
    return input0_layout.data_type == data_types::f32 && input1_layout.data_type == data_types::f32 && output_layout.data_type == data_types::f32 &&
           (fused_input_layout == nullptr || fused_input_layout->data_type == data_types::f32) && is_aligned_for_vector_width(input0_layout, width) &&
           is_aligned_for_vector_width(input1_layout, width) && is_aligned_for_vector_width(output_layout, width) &&
           (fused_input_layout == nullptr || is_aligned_for_vector_width(*fused_input_layout, width));
}

dense_vector_width select_f32_dense_vector_width(const layout& input0_layout,
                                                 const layout& input1_layout,
                                                 const layout& output_layout,
                                                 const layout* fused_input_layout,
                                                 const device_info& info) {
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0) {
        return dense_vector_width::scalar;
    }
    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    constexpr std::array candidates{dense_vector_width::vec4, dense_vector_width::vec2};
    for (const auto width : candidates) {
        if (!can_use_f32_dense_vector_width(input0_layout, input1_layout, output_layout, fused_input_layout, width)) {
            continue;
        }
        const auto vector_width = value(width);
        const auto element_count = output_layout.count();
        const auto invocation_count = element_count / vector_width + (element_count % vector_width != 0 ? 1 : 0);
        const bool wide_subgroup = subgroup_size >= subgroups_per_full_work_group * vector_width;
        const auto required_full_work_groups = wide_subgroup ? uint64_t{1} : subgroups_per_full_work_group;
        if (invocation_count / max_work_group_size >= required_full_work_groups) {
            return width;
        }
    }
    return dense_vector_width::scalar;
}

bool can_use_f32_no_tail_kernel(const layout& output_layout, dense_vector_width width, const device_info& info) {
    const auto vector_width = value(width);
    if (output_layout.count() % vector_width != 0 || info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0) {
        return false;
    }
    const auto invocation_count = checked_u32(output_layout.count() / vector_width, "F32 vector invocation count");
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    const auto local_size = select_local_work_group_size(invocation_count, max_work_group_size);
    if (invocation_count % local_size != 0) {
        return false;
    }

    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    const bool wide_subgroup = subgroup_size >= subgroups_per_full_work_group * vector_width;
    const auto full_work_groups = invocation_count / max_work_group_size;
    const auto wide_subgroup_work_group_limit = subgroups_per_full_work_group * subgroups_per_full_work_group;
    return !wide_subgroup || full_work_groups <= wide_subgroup_work_group_limit;
}

kernel_kind select_f32_vector_kernel_kind(dense_vector_width width, bool fused, bool no_tail) {
    if (width == dense_vector_width::vec4) {
        if (no_tail) {
            return fused ? kernel_kind::fused_dense_f32_vec4_no_tail_push_constants : kernel_kind::dense_f32_vec4_no_tail_push_constants;
        }
        return fused ? kernel_kind::fused_dense_f32_vec4_push_constants : kernel_kind::dense_f32_vec4_push_constants;
    }
    OPENVINO_ASSERT(width == dense_vector_width::vec2, "[GPU][Vulkan] Invalid dense vector width selected for Eltwise");
    if (no_tail) {
        return fused ? kernel_kind::fused_dense_f32_vec2_no_tail_push_constants : kernel_kind::dense_f32_vec2_no_tail_push_constants;
    }
    return fused ? kernel_kind::fused_dense_f32_vec2_push_constants : kernel_kind::dense_f32_vec2_push_constants;
}

kernel_kind select_pre_specialized_kernel_kind(kernel_kind fallback,
                                               eltwise_mode mode,
                                               const layout& input0_layout,
                                               const layout& input1_layout,
                                               const layout& output_layout) {
    if (fallback == kernel_kind::dense_push_constants && input0_layout.data_type == input1_layout.data_type &&
        input0_layout.data_type == output_layout.data_type) {
        if (input0_layout.data_type == data_types::f32) {
            if (mode == eltwise_mode::sum) {
                return kernel_kind::dense_f32_sum_push_constants;
            }
            if (mode == eltwise_mode::div) {
                return kernel_kind::dense_f32_div_push_constants;
            }
        }
        if (input0_layout.data_type == data_types::i64) {
            if (mode == eltwise_mode::sum) {
                return kernel_kind::dense_i64_sum_push_constants;
            }
            if (mode == eltwise_mode::div) {
                return kernel_kind::dense_i64_div_push_constants;
            }
        }
    }
    if (mode == eltwise_mode::eq && input0_layout.data_type == data_types::f32 && input1_layout.data_type == data_types::f32 &&
        output_layout.data_type == data_types::boolean) {
        if (fallback == kernel_kind::broadcast_scalar) {
            return kernel_kind::broadcast_f32_eq;
        }
        if (fallback == kernel_kind::broadcast_fast_scalar) {
            return kernel_kind::broadcast_fast_f32_eq;
        }
    }
    return fallback;
}

struct fused_eltwise_info {
    const fused_primitive_desc* descriptor = nullptr;
    size_t external_dependency_index = 0;
    shader_abi::fused_input_position external_position = shader_abi::fused_input_position::rhs;
    bool broadcast_input = false;
};

using fused_eltwise_chain = std::vector<fused_eltwise_info>;

struct fused_post_op_info {
    const fused_primitive_desc* descriptor = nullptr;
    shader_abi::post_op_kind kind = shader_abi::post_op_kind::activation;
    shader_abi::post_activation activation = shader_abi::post_activation::none;
    uint32_t quantize_flags = 0;
};

std::optional<shader_abi::post_activation> post_activation_code(activation_func function) {
    switch (function) {
    case activation_func::none:
        return shader_abi::post_activation::none;
    case activation_func::relu:
        return shader_abi::post_activation::relu;
    case activation_func::relu_negative_slope:
        return shader_abi::post_activation::relu_negative_slope;
    case activation_func::clamp:
        return shader_abi::post_activation::clamp;
    case activation_func::abs:
        return shader_abi::post_activation::abs;
    case activation_func::linear:
        return shader_abi::post_activation::linear;
    case activation_func::square:
        return shader_abi::post_activation::square;
    case activation_func::sqrt:
        return shader_abi::post_activation::sqrt;
    case activation_func::negative:
        return shader_abi::post_activation::negative;
    case activation_func::floor:
        return shader_abi::post_activation::floor;
    case activation_func::ceil:
        return shader_abi::post_activation::ceil;
    case activation_func::reciprocal:
        return shader_abi::post_activation::reciprocal;
    case activation_func::hard_sigmoid:
        return shader_abi::post_activation::hard_sigmoid;
    case activation_func::hsigmoid:
        return shader_abi::post_activation::hsigmoid;
    case activation_func::hswish:
        return shader_abi::post_activation::hswish;
    default:
        return std::nullopt;
    }
}

bool same_dense_geometry(const layout& input_layout, const layout& output_layout) {
    return !input_layout.is_dynamic() && !output_layout.is_dynamic() &&
           has_dense_storage(input_layout) && has_dense_storage(output_layout) &&
           input_layout.get_shape() == output_layout.get_shape() &&
           input_layout.format == output_layout.format &&
           input_layout.data_padding == output_layout.data_padding;
}

uint32_t post_quantize_flags(const QuantizeFuseParams& params) {
    uint32_t flags = 0;
    flags |= params._need_pre_shift
                 ? shader_abi::value(shader_abi::post_quantize_flag::need_pre_shift)
                 : 0U;
    flags |= params._need_post_scale
                 ? shader_abi::value(shader_abi::post_quantize_flag::need_post_scale)
                 : 0U;
    flags |= params._need_post_shift
                 ? shader_abi::value(shader_abi::post_quantize_flag::need_post_shift)
                 : 0U;
    flags |= params._need_clamp
                 ? shader_abi::value(shader_abi::post_quantize_flag::need_clamp)
                 : 0U;
    flags |= params._need_min_clamp
                 ? shader_abi::value(shader_abi::post_quantize_flag::need_min_clamp)
                 : 0U;
    flags |= params._need_max_clamp
                 ? shader_abi::value(shader_abi::post_quantize_flag::need_max_clamp)
                 : 0U;
    flags |= params._per_tensor_output_range && params._out_lo < params._out_hi
                 ? shader_abi::value(shader_abi::post_quantize_flag::use_output_range)
                 : 0U;
    return flags;
}

std::optional<fused_post_op_info> get_supported_fused_post_op(const program_node& node) {
    const auto& fused_primitives = node.get_fused_primitives();
    if (fused_primitives.size() != 1) {
        return std::nullopt;
    }
    const auto& fused = fused_primitives.front();
    const auto& input_layout = fused.input_layout;
    const auto& output_layout = fused.output_layout;
    if (!fused.deps.empty() || fused.total_num_deps != 1 ||
        !same_dense_geometry(input_layout, output_layout) ||
        !output_layout.identical(node.get_output_layout(0)) ||
        !one_of(input_layout.data_type, {data_types::f16, data_types::f32}) ||
        !is_supported_data_type(output_layout.data_type) ||
        !is_supported_format(output_layout.format.value) || output_layout.get_rank() > max_rank) {
        return std::nullopt;
    }

    fused_post_op_info result;
    result.descriptor = &fused;
    if (fused.is_type<activation>()) {
        const auto params = fused.get_typed_fuse_params<ActivationFuseParams>();
        const auto activation = post_activation_code(params->_desc->activation_function);
        if (!activation.has_value() || params->_desc->additional_params_input.is_valid() ||
            input_layout.data_type != output_layout.data_type) {
            return std::nullopt;
        }
        result.kind = shader_abi::post_op_kind::activation;
        result.activation = *activation;
        return result;
    }
    if (fused.is_type<quantize>()) {
        const auto params = fused.get_typed_fuse_params<QuantizeFuseParams>();
        const bool all_values_are_per_tensor =
            params->_per_tensor_input_range && params->_per_tensor_input_scale &&
            (!params->_need_pre_shift || params->_per_tensor_input_shift) &&
            params->_per_tensor_output_range &&
            (!params->_need_post_scale || params->_per_tensor_output_scale) &&
            (!params->_need_post_shift || params->_per_tensor_output_shift);
        if (!params->_scale_shift_opt || !all_values_are_per_tensor ||
            !one_of(output_layout.data_type,
                    {data_types::f16, data_types::f32, data_types::i8, data_types::u8})) {
            return std::nullopt;
        }
        result.kind = shader_abi::post_op_kind::quantize;
        result.quantize_flags = post_quantize_flags(*params);
        return result;
    }
    if (fused.is_type<reorder>()) {
        const auto desc = fused.typed_desc<reorder>();
        if (!desc->truncate || input_layout.data_type == output_layout.data_type ||
            !one_of(output_layout.data_type, {data_types::f16, data_types::f32})) {
            return std::nullopt;
        }
        result.kind = shader_abi::post_op_kind::convert;
        return result;
    }
    return std::nullopt;
}

bool can_use_fused_post_op_kernel(const layout& input0_layout,
                                  const layout& input1_layout,
                                  const fused_post_op_info& post_op,
                                  const layout& output_layout) {
    const auto& base_output_layout = post_op.descriptor->input_layout;
    if (!input0_layout.identical(base_output_layout) || !input1_layout.identical(base_output_layout) ||
        !same_dense_geometry(base_output_layout, output_layout)) {
        return false;
    }
    const auto packed_width = output_elements_per_invocation(output_layout);
    return packed_width == 1 || output_layout.get_linear_offset() % packed_width == 0;
}

bool is_numpy_broadcast_compatible(const layout& input_layout, const layout& output_layout) {
    if (input_layout.is_dynamic() || output_layout.is_dynamic() || input_layout.get_rank() > output_layout.get_rank()) {
        return false;
    }
    const auto input_shape = input_layout.get_shape();
    const auto output_shape = output_layout.get_shape();
    const auto leading_dimensions = output_shape.size() - input_shape.size();
    for (size_t input_axis = 0; input_axis < input_shape.size(); ++input_axis) {
        const auto input_dimension = input_shape[input_axis];
        const auto output_dimension = output_shape[leading_dimensions + input_axis];
        if (input_dimension != 1 && input_dimension != output_dimension) {
            return false;
        }
    }
    return true;
}

std::optional<fused_eltwise_chain> get_supported_fused_eltwise_chain(const program_node& node) {
    const auto& fused_primitives = node.get_fused_primitives();
    if (fused_primitives.empty() || fused_primitives.size() > max_fused_chain_length) {
        return std::nullopt;
    }

    const auto& output_layout = node.get_output_layout(0);
    if (output_layout.is_dynamic() || !has_dense_storage(output_layout) || node.get_dependencies().size() != 2 + fused_primitives.size()) {
        return std::nullopt;
    }
    for (size_t input_index : {size_t{0}, size_t{1}}) {
        const auto& input_layout = node.get_input_layout(input_index);
        if (input_layout.is_dynamic() || !input_layout.identical(output_layout) || !has_dense_storage(input_layout)) {
            return std::nullopt;
        }
    }

    fused_eltwise_chain chain;
    chain.reserve(fused_primitives.size());
    for (size_t stage = 0; stage < fused_primitives.size(); ++stage) {
        const auto& fused = fused_primitives[stage];
        if (!fused.is_type<eltwise>()) {
            return std::nullopt;
        }
        const auto fused_desc = fused.typed_desc<eltwise>();
        const bool coefficients_supported =
            fused_desc->coefficients.empty() || (fused_desc->mode == eltwise_mode::sum && fused_desc->coefficients.size() == 2);
        if (!is_fused_mode(fused_desc->mode) || !fused_desc->stride.empty() || !coefficients_supported || fused.inputs.size() != 2 ||
            fused.total_num_deps != 2 || fused.deps.size() != 1 || !fused.has_outer_dep()) {
            return std::nullopt;
        }

        std::optional<size_t> external_position;
        std::optional<size_t> internal_position;
        for (size_t position = 0; position < fused.inputs.size(); ++position) {
            const auto& input = fused.inputs[position];
            if (input.m_type == FusedInputType::EXTERNAL) {
                if (external_position.has_value() || input.m_idx >= node.get_dependencies().size()) {
                    return std::nullopt;
                }
                external_position = position;
            } else if ((stage == 0 && (input.m_type != FusedInputType::ORIGINAL || input.m_idx != 0)) ||
                       (stage != 0 && (input.m_type != FusedInputType::INTERNAL || input.m_idx != stage - 1))) {
                return std::nullopt;
            }
            if (input.m_type != FusedInputType::EXTERNAL) {
                if (internal_position.has_value()) {
                    return std::nullopt;
                }
                internal_position = position;
            }
        }
        if (!external_position.has_value() || !internal_position.has_value()) {
            return std::nullopt;
        }

        const auto external_dependency_index = fused.inputs[*external_position].m_idx;
        const auto expected_external_dependency_index = 2 + stage;
        if (external_dependency_index != expected_external_dependency_index ||
            external_dependency_index != static_cast<size_t>(fused.outer_dep_start_idx)) {
            return std::nullopt;
        }
        const auto& input_layout = node.get_input_layout(external_dependency_index);
        const bool broadcast_input = !input_layout.identical(output_layout);
        if (input_layout.is_dynamic() || !has_dense_storage(input_layout) ||
            !is_supported_data_type(input_layout.data_type) ||
            !is_supported_format(input_layout.format.value) || input_layout.get_rank() > max_rank ||
            (broadcast_input &&
             (fused_primitives.size() != 1 ||
              data_type_traits::size_of(output_layout.data_type) < sizeof(uint32_t) ||
              !is_numpy_broadcast_compatible(input_layout, output_layout)))) {
            return std::nullopt;
        }
        if (!broadcast_input &&
            !can_use_fused_dense_kernel(node.get_input_layout(0),
                                        node.get_input_layout(1),
                                        input_layout,
                                        output_layout)) {
            return std::nullopt;
        }

        chain.push_back(fused_eltwise_info{&fused,
                                           external_dependency_index,
                                           *external_position == 0 ? shader_abi::fused_input_position::lhs
                                                                   : shader_abi::fused_input_position::rhs,
                                           broadcast_input});
    }

    return chain;
}

uint32_t operation_batch_width_limit(eltwise_mode mode, data_types type, kernel_kind kind, const device_info& info) {
    switch (mode) {
    case eltwise_mode::pow:
    case eltwise_mode::atan2:
        return scalar_batch_width;
    case eltwise_mode::div:
    case eltwise_mode::mod:
    case eltwise_mode::floor_mod:
        if (type == data_types::i64) {
            return scalar_batch_width;
        }
        if (is_broadcast_vector_kernel(kind) && info.max_work_group_size != 0) {
            const auto capability_width = std::max<uint64_t>(info.max_work_group_size / portable_max_local_work_group_size, scalar_batch_width);
            return static_cast<uint32_t>(std::min<uint64_t>(capability_width, balanced_scalar_batch_width));
        }
        return division_scalar_batch_width;
    case eltwise_mode::squared_diff:
        return balanced_scalar_batch_width;
    default:
        return maximum_scalar_batch_width;
    }
}

bool has_aligned_offset(const layout& tensor_layout, uint32_t width) {
    return tensor_layout.get_linear_offset() % width == 0;
}

uint32_t select_generic_elements_per_invocation(kernel_kind kind,
                                                const layout& input0_layout,
                                                const layout& input1_layout,
                                                const layout& output_layout,
                                                const layout* fused_input_layout,
                                                eltwise_mode mode,
                                                const std::vector<eltwise_mode>& fused_modes,
                                                const device_info& info) {
    const auto packed_width = get_packed_dense_width(kind);
    if (packed_width != packed_dense_width::none) {
        return value(packed_width);
    }
    const auto vector_width = get_dense_vector_width(kind);
    if (vector_width != dense_vector_width::scalar) {
        return value(vector_width);
    }
    const auto output_width = output_elements_per_invocation(output_layout);
    if (output_width != 1 || kind == kernel_kind::unary || kind == kernel_kind::broadcast_scalar || kind == kernel_kind::broadcast_fast_scalar) {
        return output_width;
    }
    OPENVINO_ASSERT(is_plain_dense_kernel(kind) || is_fused_kernel(kind) ||
                        is_fused_post_op_kernel(kind) ||
                        is_broadcast_vector_kernel(kind) || kind == kernel_kind::scalar_constant,
                    "[GPU][Vulkan] Eltwise scalar batch selection received an unsupported kernel kind");

    uint32_t width_limit = std::min({operation_batch_width_limit(mode, input0_layout.data_type, kind, info),
                                     operation_batch_width_limit(mode, input1_layout.data_type, kind, info),
                                     operation_batch_width_limit(mode, output_layout.data_type, kind, info)});
    for (const auto fused_mode : fused_modes) {
        width_limit = std::min(width_limit, operation_batch_width_limit(fused_mode, output_layout.data_type, kind, info));
    }
    if (is_plain_dense_kernel(kind) || is_fused_kernel(kind) ||
        is_fused_post_op_kernel(kind) || kind == kernel_kind::scalar_constant) {
        width_limit = std::min(width_limit, balanced_scalar_batch_width);
    }
    auto maximum_scalar_size = std::max({data_type_traits::size_of(input0_layout.data_type),
                                         data_type_traits::size_of(input1_layout.data_type),
                                         data_type_traits::size_of(output_layout.data_type)});
    if (fused_input_layout != nullptr) {
        maximum_scalar_size = std::max(maximum_scalar_size, data_type_traits::size_of(fused_input_layout->data_type));
    }
    if (maximum_scalar_size >= sizeof(uint64_t) && width_limit >= division_scalar_batch_width) {
        return division_scalar_batch_width;
    }
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0 || info.max_work_group_size == 0) {
        return scalar_batch_width;
    }

    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroup_width_limit = std::max<uint64_t>(scalar_elements_per_subgroup_budget / subgroup_size, balanced_scalar_batch_width);
    width_limit = static_cast<uint32_t>(std::min<uint64_t>(width_limit, subgroup_width_limit));
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    const auto useful_subgroup_coverage = std::min<uint64_t>(subgroups_per_full_work_group, width_limit);
    const auto required_parallel_invocations = max_work_group_size * useful_subgroup_coverage;
    const auto has_aligned_layouts = [&](uint32_t candidate) {
        if (!has_aligned_offset(output_layout, candidate)) {
            return false;
        }
        return !(is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind)) ||
               (has_aligned_offset(input0_layout, candidate) && has_aligned_offset(input1_layout, candidate) &&
                (fused_input_layout == nullptr || is_fused_broadcast_kernel(kind) ||
                 has_aligned_offset(*fused_input_layout, candidate)));
    };
    constexpr std::array candidates{maximum_scalar_batch_width, balanced_scalar_batch_width, division_scalar_batch_width, scalar_batch_width};
    for (const auto candidate : candidates) {
        if (candidate > width_limit || !has_aligned_layouts(candidate)) {
            continue;
        }
        const auto invocation_count = (output_layout.count() + candidate - 1) / candidate;
        if (invocation_count >= required_parallel_invocations) {
            return candidate;
        }
    }
    return scalar_batch_width;
}

bool can_use_broadcast_vector_kernel(const layout& output_layout) {
    if (data_type_traits::size_of(output_layout.data_type) < sizeof(uint32_t) || output_layout.count() < broadcast_vector_min_elements) {
        return false;
    }
    return output_layout.count() % broadcast_vector_width == 0;
}

void write_tensor_metadata_at(std::array<uint32_t, metadata_words>& metadata,
                              uint32_t base,
                              const layout& tensor_layout,
                              uint32_t output_rank) {
    const auto shape = tensor_layout.get_shape();
    const auto pitches = tensor_layout.get_pitches();
    OPENVINO_ASSERT(shape.size() <= pitches.size() && shape.size() <= output_rank, "[GPU][Vulkan] Eltwise received an invalid tensor rank");

    const auto leading_dimensions = output_rank - checked_u32(shape.size(), "rank");
    for (uint32_t axis = 0; axis < max_rank; ++axis) {
        metadata[base + axis] = 1;
        metadata[base + max_rank + axis] = 0;
    }
    for (uint32_t axis = 0; axis < shape.size(); ++axis) {
        const auto output_axis = leading_dimensions + axis;
        metadata[base + output_axis] = checked_u32(shape[axis], "dimension");
        metadata[base + max_rank + output_axis] = checked_u32(pitches[axis], "pitch");
    }
    metadata[base + max_rank * 2] = checked_u32(tensor_layout.get_linear_offset(), "base offset");
}

void write_tensor_metadata(std::array<uint32_t, metadata_words>& metadata,
                           shader_abi::tensor_index tensor,
                           const layout& tensor_layout,
                           uint32_t output_rank) {
    write_tensor_metadata_at(metadata,
                             header_words + shader_abi::index(tensor) * tensor_words,
                             tensor_layout,
                             output_rank);
}

bool collapsed_tensor_axis(const std::array<uint32_t, metadata_words>& metadata,
                           uint32_t base,
                           uint32_t axis,
                           uint32_t output_left_dimension,
                           uint32_t output_right_dimension,
                           uint32_t& collapsed_dimension,
                           uint32_t& collapsed_pitch) {
    const auto left_dimension = metadata[base + axis];
    const auto right_dimension = metadata[base + axis + 1];
    const auto left_pitch = metadata[base + max_rank + axis];
    const auto right_pitch = metadata[base + max_rank + axis + 1];

    if (output_left_dimension == 1) {
        collapsed_dimension = right_dimension;
        collapsed_pitch = right_pitch;
        return true;
    }
    if (output_right_dimension == 1) {
        collapsed_dimension = left_dimension;
        collapsed_pitch = left_pitch;
        return true;
    }
    if (left_dimension == 1 && right_dimension == 1) {
        collapsed_dimension = 1;
        collapsed_pitch = 0;
        return true;
    }
    if (left_dimension == output_left_dimension && right_dimension == output_right_dimension &&
        static_cast<uint64_t>(left_pitch) == static_cast<uint64_t>(right_pitch) * right_dimension) {
        collapsed_dimension = checked_u32(static_cast<uint64_t>(left_dimension) * right_dimension, "collapsed dimension");
        collapsed_pitch = right_pitch;
        return true;
    }
    return false;
}

uint32_t collapse_metadata_dimensions(std::array<uint32_t, metadata_words>& metadata,
                                      uint32_t rank,
                                      std::optional<uint32_t> extra_tensor_base = std::nullopt) {
    if (rank < 2) {
        return rank;
    }

    for (int32_t axis = static_cast<int32_t>(rank) - 2; axis >= 0; --axis) {
        const auto output_base = header_words + shader_abi::index(shader_abi::tensor_index::output) * tensor_words;
        const auto output_left_dimension = metadata[output_base + static_cast<uint32_t>(axis)];
        const auto output_right_dimension = metadata[output_base + static_cast<uint32_t>(axis) + 1];
        std::array<uint32_t, tensor_count + 1> tensor_bases{};
        for (uint32_t tensor = 0; tensor < tensor_count; ++tensor) {
            tensor_bases[tensor] = header_words + tensor * tensor_words;
        }
        const auto active_tensor_count = tensor_count + (extra_tensor_base.has_value() ? 1U : 0U);
        if (extra_tensor_base.has_value()) {
            tensor_bases[tensor_count] = *extra_tensor_base;
        }
        std::array<uint32_t, tensor_count + 1> collapsed_dimensions{};
        std::array<uint32_t, tensor_count + 1> collapsed_pitches{};
        bool can_collapse = true;
        for (uint32_t tensor = 0; tensor < active_tensor_count; ++tensor) {
            can_collapse &= collapsed_tensor_axis(metadata,
                                                  tensor_bases[tensor],
                                                  static_cast<uint32_t>(axis),
                                                  output_left_dimension,
                                                  output_right_dimension,
                                                  collapsed_dimensions[tensor],
                                                  collapsed_pitches[tensor]);
        }
        if (!can_collapse) {
            continue;
        }

        for (uint32_t tensor = 0; tensor < active_tensor_count; ++tensor) {
            const auto base = tensor_bases[tensor];
            metadata[base + static_cast<uint32_t>(axis)] = collapsed_dimensions[tensor];
            metadata[base + max_rank + static_cast<uint32_t>(axis)] = collapsed_pitches[tensor];
            for (uint32_t shifted_axis = static_cast<uint32_t>(axis) + 1; shifted_axis + 1 < rank; ++shifted_axis) {
                metadata[base + shifted_axis] = metadata[base + shifted_axis + 1];
                metadata[base + max_rank + shifted_axis] = metadata[base + max_rank + shifted_axis + 1];
            }
            metadata[base + rank - 1] = 1;
            metadata[base + max_rank + rank - 1] = 0;
        }
        --rank;
    }
    return rank;
}

uint32_t fast_divisor_reciprocal(uint32_t divisor) {
    OPENVINO_ASSERT(divisor != 0, "[GPU][Vulkan] Eltwise broadcast divisor must be non-zero");
    if (divisor == 1) {
        return 0;
    }
    return static_cast<uint32_t>((uint64_t{1} << 32) / divisor);
}

void write_fast_divisor_metadata(std::array<uint32_t, metadata_words>& metadata, bool fused_chain) {
    const auto rank = metadata[shader_abi::index(shader_abi::metadata_field::rank)];
    const auto output_base = header_words + shader_abi::index(shader_abi::tensor_index::output) * tensor_words;
    const auto metadata_base = fused_chain ? fused_chain_fast_divisor_metadata_base : regular_fast_divisor_metadata_base;
    for (uint32_t axis = 0; axis < max_rank; ++axis) {
        const auto dimension = axis < rank ? metadata[output_base + axis] : 1U;
        metadata[metadata_base + axis] = fast_divisor_reciprocal(dimension);
    }
}

uint32_t active_broadcast_axes(const std::array<uint32_t, metadata_words>& metadata, shader_abi::tensor_index tensor) {
    const auto rank = metadata[shader_abi::index(shader_abi::metadata_field::rank)];
    const auto base = header_words + shader_abi::index(tensor) * tensor_words;
    uint32_t axes = 0;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (metadata[base + axis] != 1 && metadata[base + max_rank + axis] != 0) {
            axes |= 1U << axis;
        }
    }
    return axes;
}

uint32_t collapsed_broadcast_rank(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    const auto output_rank = checked_u32(output_layout.get_shape().size(), "output rank");
    std::array<uint32_t, metadata_words> metadata{};
    write_tensor_metadata(metadata, shader_abi::tensor_index::input0, input0_layout, output_rank);
    write_tensor_metadata(metadata, shader_abi::tensor_index::input1, input1_layout, output_rank);
    write_tensor_metadata(metadata, shader_abi::tensor_index::output, output_layout, output_rank);
    return collapse_metadata_dimensions(metadata, output_rank);
}

bool should_use_fast_broadcast_kernel(const layout& input0_layout,
                                      const layout& input1_layout,
                                      const layout& output_layout,
                                      const device_info& info,
                                      uint32_t elements_per_invocation) {
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0 || info.max_work_group_size == 0) {
        return false;
    }
    const auto subgroup_size = info.supported_simd_sizes.front();
    const auto subgroup_slots = std::max<uint64_t>(info.max_work_group_size / subgroup_size, 1);
    const auto rank = collapsed_broadcast_rank(input0_layout, input1_layout, output_layout);
    const auto coordinate_schedule_pressure = static_cast<uint64_t>(rank) * subgroup_slots;
    const auto invocation_count = (output_layout.count() + elements_per_invocation - 1) / elements_per_invocation;
    return coordinate_schedule_pressure <= portable_max_local_work_group_size && invocation_count >= subgroup_size;
}

std::array<uint32_t, metadata_words> make_metadata(const eltwise_inst& instance,
                                                   const std::optional<scalar_constant>& scalar,
                                                   const std::optional<fused_eltwise_chain>& fused,
                                                   const std::optional<fused_post_op_info>& post_op) {
    const auto desc = instance.get_typed_desc<eltwise>();
    const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
    const auto fused_input_count = fused.has_value() ? fused->size() : 0U;
    OPENVINO_ASSERT(instance.inputs_memory_count() == base_input_count && instance.get_fused_mem_count() == fused_input_count,
                    "[GPU][Vulkan] Eltwise received an unexpected regular or fused input count");
    const auto& input0_layout = instance.get_input_layout(0);
    const auto& input1_layout = base_input_count == 1 ? input0_layout : instance.get_input_layout(1);
    const auto& output_layout = instance.get_output_layout(0);
    OPENVINO_ASSERT(!input0_layout.is_dynamic() && !input1_layout.is_dynamic() && !output_layout.is_dynamic(),
                    "[GPU][Vulkan] Eltwise execution requires resolved runtime layouts");
    const auto output_rank = checked_u32(output_layout.get_shape().size(), "output rank");
    OPENVINO_ASSERT(output_rank <= max_rank, "[GPU][Vulkan] Eltwise supports tensors with rank up to ", max_rank);

    std::array<uint32_t, metadata_words> metadata{};
    metadata[shader_abi::index(shader_abi::metadata_field::element_count)] = checked_u32(output_layout.count(), "element count");
    metadata[shader_abi::index(shader_abi::metadata_field::mode)] = shader_abi::value(shader_mode_code(desc->mode));
    metadata[shader_abi::index(shader_abi::metadata_field::rank)] = output_rank;
    const auto scalar_tensor_index =
        scalar.has_value() && scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::tensor_index::input1 : shader_abi::tensor_index::input0;
    const auto& scalar_tensor_layout = scalar_tensor_index == shader_abi::tensor_index::input0 ? input0_layout : input1_layout;
    metadata[shader_abi::index(shader_abi::metadata_field::flags)] =
        ((scalar.has_value() ? can_use_scalar_linear_storage(scalar_tensor_layout, output_layout)
                             : can_use_linear_storage(input0_layout, input1_layout, output_layout))
             ? shader_abi::value(shader_abi::storage_flag::linear)
             : 0U) |
        (output_elements_per_invocation(output_layout) > 1 ? shader_abi::value(shader_abi::storage_flag::packed_output) : 0U) |
        (scalar.has_value() && scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::value(shader_abi::storage_flag::scalar_constant_input0)
                                                                                       : 0U) |
        (fused.has_value() && fused->size() == 1 && fused->front().broadcast_input &&
                 instance.get_input_layout(fused->front().external_dependency_index).count() == 1
             ? shader_abi::value(shader_abi::storage_flag::fused_scalar_input)
             : 0U);
    metadata[shader_abi::index(shader_abi::metadata_field::input0_type)] = shader_abi::value(scalar_type_code(input0_layout.data_type));
    metadata[shader_abi::index(shader_abi::metadata_field::input1_type)] = shader_abi::value(scalar_type_code(input1_layout.data_type));
    metadata[shader_abi::index(shader_abi::metadata_field::output_type)] = shader_abi::value(scalar_type_code(output_layout.data_type));
    metadata[shader_abi::index(shader_abi::metadata_field::python_division)] = desc->m_pythondiv ? 1U : 0U;
    metadata[shader_abi::index(shader_abi::metadata_field::infinity_detection)] =
        desc->mode == eltwise_mode::is_inf && desc->coefficients.size() == 2
            ? (desc->coefficients[0] != 0.0f ? shader_abi::value(shader_abi::infinity_flag::negative) : 0U) |
                  (desc->coefficients[1] != 0.0f ? shader_abi::value(shader_abi::infinity_flag::positive) : 0U)
            : shader_abi::all_infinities_mask;
    metadata[shader_abi::index(shader_abi::metadata_field::input_count)] = base_input_count;
    metadata[shader_abi::index(shader_abi::metadata_field::input0_coefficient)] = float_bits(desc->coefficients.empty() ? 1.0f : desc->coefficients[0]);
    metadata[shader_abi::index(shader_abi::metadata_field::input1_coefficient)] = float_bits(desc->coefficients.empty() ? 1.0f : desc->coefficients[1]);
    write_tensor_metadata(metadata, shader_abi::tensor_index::input0, input0_layout, output_rank);
    write_tensor_metadata(metadata, shader_abi::tensor_index::input1, input1_layout, output_rank);
    write_tensor_metadata(metadata, shader_abi::tensor_index::output, output_layout, output_rank);
    std::optional<uint32_t> extra_tensor_base;
    if (fused.has_value() && fused->size() == 1 && fused->front().broadcast_input) {
        write_tensor_metadata_at(metadata,
                                 fused_broadcast_metadata_base,
                                 instance.get_input_layout(fused->front().external_dependency_index),
                                 output_rank);
        extra_tensor_base = fused_broadcast_metadata_base;
    }
    metadata[shader_abi::index(shader_abi::metadata_field::rank)] =
        collapse_metadata_dimensions(metadata, output_rank, extra_tensor_base);
    write_fast_divisor_metadata(metadata, fused.has_value() && fused->size() > 1);
    if (scalar.has_value()) {
        const auto scalar_metadata_base = header_words + shader_abi::index(scalar->input_index) * tensor_words;
        metadata[scalar_metadata_base] = scalar->bits[0];
        metadata[scalar_metadata_base + max_rank] = scalar->bits[1];
    }
    if (fused.has_value()) {
        metadata[fused_chain_metadata_base + shader_abi::index(shader_abi::fused_chain_metadata_field::chain_length)] =
            checked_u32(fused->size(), "fused Eltwise chain length");
        for (size_t stage = 0; stage < fused->size(); ++stage) {
            const auto& fused_stage = fused->at(stage);
            const auto fused_desc = fused_stage.descriptor->typed_desc<eltwise>();
            const auto& fused_input_layout = instance.get_input_layout(fused_stage.external_dependency_index);
            const auto stage_metadata_base = fused_metadata_base + stage * fused_metadata_words;
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::mode)] =
                shader_abi::value(shader_mode_code(fused_desc->mode));
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_type)] =
                shader_abi::value(scalar_type_code(fused_input_layout.data_type));
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_position)] =
                shader_abi::value(fused_stage.external_position);
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::python_division)] = fused_desc->m_pythondiv ? 1U : 0U;
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input0_coefficient)] =
                float_bits(fused_desc->coefficients.empty() ? 1.0f : fused_desc->coefficients[0]);
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input1_coefficient)] =
                float_bits(fused_desc->coefficients.empty() ? 1.0f : fused_desc->coefficients[1]);
            metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_offset)] =
                checked_u32(fused_input_layout.get_linear_offset(), "fused input base offset");
        }
    }
    if (post_op.has_value()) {
        const auto write_post_value = [&](shader_abi::post_op_metadata_field field, float value) {
            metadata[post_op_metadata_base + shader_abi::index(field)] = float_bits(value);
        };
        if (post_op->kind == shader_abi::post_op_kind::activation) {
            const auto params = post_op->descriptor->get_typed_fuse_params<ActivationFuseParams>();
            write_post_value(shader_abi::post_op_metadata_field::parameter_a,
                             params->_desc->additional_params.a);
            write_post_value(shader_abi::post_op_metadata_field::parameter_b,
                             params->_desc->additional_params.b);
        } else if (post_op->kind == shader_abi::post_op_kind::quantize) {
            const auto params = post_op->descriptor->get_typed_fuse_params<QuantizeFuseParams>();
            write_post_value(shader_abi::post_op_metadata_field::input_low, params->_in_lo);
            write_post_value(shader_abi::post_op_metadata_field::input_high, params->_in_hi);
            write_post_value(shader_abi::post_op_metadata_field::input_scale, params->_in_scale);
            write_post_value(shader_abi::post_op_metadata_field::input_shift, params->_in_shift);
            write_post_value(shader_abi::post_op_metadata_field::output_low, params->_out_lo);
            write_post_value(shader_abi::post_op_metadata_field::output_high, params->_out_hi);
            write_post_value(shader_abi::post_op_metadata_field::output_scale, params->_out_scale);
            write_post_value(shader_abi::post_op_metadata_field::output_shift, params->_out_shift);
        }
    }
    return metadata;
}

void append_u32_scalar(scalars_desc& scalars, uint32_t value) {
    scalar_desc scalar{};
    scalar.t = scalar_desc::Types::UINT32;
    scalar.v.u32 = value;
    scalars.push_back(scalar);
}

scalars_desc make_dense_push_constants(const std::array<uint32_t, metadata_words>& metadata, bool fused) {
    scalars_desc result;
    result.reserve(fused ? fused_dense_push_constant_words : dense_push_constant_words);
    const auto tensor_offset = [&metadata](shader_abi::tensor_index tensor) {
        return metadata[header_words + shader_abi::index(tensor) * tensor_words + max_rank * 2];
    };
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::element_count)]);
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::input0));
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::input1));
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::output));
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::python_division)]);
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::infinity_detection)]);
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::input0_coefficient)]);
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::input1_coefficient)]);
    if (fused) {
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_offset)]);
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::python_division)]);
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input0_coefficient)]);
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input1_coefficient)]);
    }
    return result;
}

struct spirv_blob {
    const uint32_t* data = nullptr;
    size_t size = 0;
};

std::shared_ptr<kernel_string> make_kernel_source(kernel_kind kind, bool restrict_output, uint32_t fused_chain_length) {
    OPENVINO_ASSERT(!restrict_output || supports_restricted_output(kind),
                    "[GPU][Vulkan] Restricted Eltwise module requires the single-output dense ABI");
    auto source = std::make_shared<kernel_string>();
    const uint32_t* spirv = nullptr;
    size_t spirv_size = 0;
    const auto select_spirv = [&](const auto& alias_safe, const auto& restricted) {
        if (restrict_output) {
            spirv = restricted;
            spirv_size = sizeof(restricted);
        } else {
            spirv = alias_safe;
            spirv_size = sizeof(alias_safe);
        }
    };
    const auto select_fused_chain_spirv = [&](const auto& alias_safe_variants, const auto& restricted_variants) {
        OPENVINO_ASSERT(fused_chain_length >= min_multi_stage_fused_chain_length && fused_chain_length <= max_fused_chain_length,
                        "[GPU][Vulkan] Invalid fixed Eltwise chain length");
        const auto variant_index = fused_chain_length - min_multi_stage_fused_chain_length;
        const auto& selected = restrict_output ? restricted_variants.at(variant_index) : alias_safe_variants.at(variant_index);
        spirv = selected.data;
        spirv_size = selected.size;
    };
    if (kind == kernel_kind::dense) {
        select_spirv(eltwise_dense_spirv, eltwise_dense_restrict_spirv);
    } else if (kind == kernel_kind::dense_push_constants) {
        select_spirv(eltwise_dense_push_constants_spirv, eltwise_dense_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_sum_push_constants) {
        select_spirv(eltwise_dense_f32_sum_push_constants_spirv, eltwise_dense_f32_sum_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_div_push_constants) {
        select_spirv(eltwise_dense_f32_div_push_constants_spirv, eltwise_dense_f32_div_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_i64_sum_push_constants) {
        select_spirv(eltwise_dense_i64_sum_push_constants_spirv, eltwise_dense_i64_sum_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_i64_div_push_constants) {
        select_spirv(eltwise_dense_i64_div_push_constants_spirv, eltwise_dense_i64_div_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec2_push_constants) {
        select_spirv(eltwise_dense_f32_vec2_push_constants_spirv, eltwise_dense_f32_vec2_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec4_push_constants) {
        select_spirv(eltwise_dense_f32_vec4_push_constants_spirv, eltwise_dense_f32_vec4_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec2_no_tail_push_constants) {
        select_spirv(eltwise_dense_f32_vec2_no_tail_push_constants_spirv, eltwise_dense_f32_vec2_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec4_no_tail_push_constants) {
        select_spirv(eltwise_dense_f32_vec4_no_tail_push_constants_spirv, eltwise_dense_f32_vec4_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_packed_8bit_push_constants) {
        select_spirv(eltwise_dense_packed_8bit_push_constants_spirv, eltwise_dense_packed_8bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_packed_16bit_push_constants) {
        select_spirv(eltwise_dense_packed_16bit_push_constants_spirv, eltwise_dense_packed_16bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_packed_f16_push_constants) {
        select_spirv(eltwise_dense_packed_f16_push_constants_spirv, eltwise_dense_packed_f16_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense) {
        select_spirv(eltwise_fused_dense_spirv, eltwise_fused_dense_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_push_constants) {
        select_spirv(eltwise_fused_dense_push_constants_spirv, eltwise_fused_dense_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec2_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec2_push_constants_spirv, eltwise_fused_dense_f32_vec2_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec4_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec4_push_constants_spirv, eltwise_fused_dense_f32_vec4_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec2_no_tail_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv, eltwise_fused_dense_f32_vec2_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec4_no_tail_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv, eltwise_fused_dense_f32_vec4_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_8bit_push_constants) {
        select_spirv(eltwise_fused_dense_packed_8bit_push_constants_spirv, eltwise_fused_dense_packed_8bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_16bit_push_constants) {
        select_spirv(eltwise_fused_dense_packed_16bit_push_constants_spirv, eltwise_fused_dense_packed_16bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_f16_push_constants) {
        select_spirv(eltwise_fused_dense_packed_f16_push_constants_spirv, eltwise_fused_dense_packed_f16_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_broadcast) {
        spirv = eltwise_fused_broadcast_spirv;
        spirv_size = sizeof(eltwise_fused_broadcast_spirv);
    } else if (kind == kernel_kind::fused_post_op) {
        spirv = eltwise_fused_post_op_spirv;
        spirv_size = sizeof(eltwise_fused_post_op_spirv);
    } else if (kind == kernel_kind::fused_dense_chain || kind == kernel_kind::fused_dense_chain_f32_vec4_no_tail) {
        constexpr size_t fixed_chain_variant_count = max_fused_chain_length - min_multi_stage_fused_chain_length + 1;
        static const std::array<spirv_blob, fixed_chain_variant_count> scalar_alias_safe{{
            {eltwise_fused_dense_chain_length2_spirv, sizeof(eltwise_fused_dense_chain_length2_spirv)},
            {eltwise_fused_dense_chain_length3_spirv, sizeof(eltwise_fused_dense_chain_length3_spirv)},
            {eltwise_fused_dense_chain_length4_spirv, sizeof(eltwise_fused_dense_chain_length4_spirv)},
            {eltwise_fused_dense_chain_length5_spirv, sizeof(eltwise_fused_dense_chain_length5_spirv)},
            {eltwise_fused_dense_chain_length6_spirv, sizeof(eltwise_fused_dense_chain_length6_spirv)},
            {eltwise_fused_dense_chain_length7_spirv, sizeof(eltwise_fused_dense_chain_length7_spirv)},
            {eltwise_fused_dense_chain_length8_spirv, sizeof(eltwise_fused_dense_chain_length8_spirv)},
        }};
        static const std::array<spirv_blob, fixed_chain_variant_count> scalar_restricted{{
            {eltwise_fused_dense_chain_length2_restrict_spirv, sizeof(eltwise_fused_dense_chain_length2_restrict_spirv)},
            {eltwise_fused_dense_chain_length3_restrict_spirv, sizeof(eltwise_fused_dense_chain_length3_restrict_spirv)},
            {eltwise_fused_dense_chain_length4_restrict_spirv, sizeof(eltwise_fused_dense_chain_length4_restrict_spirv)},
            {eltwise_fused_dense_chain_length5_restrict_spirv, sizeof(eltwise_fused_dense_chain_length5_restrict_spirv)},
            {eltwise_fused_dense_chain_length6_restrict_spirv, sizeof(eltwise_fused_dense_chain_length6_restrict_spirv)},
            {eltwise_fused_dense_chain_length7_restrict_spirv, sizeof(eltwise_fused_dense_chain_length7_restrict_spirv)},
            {eltwise_fused_dense_chain_length8_restrict_spirv, sizeof(eltwise_fused_dense_chain_length8_restrict_spirv)},
        }};
        static const std::array<spirv_blob, fixed_chain_variant_count> vector_alias_safe{{
            {eltwise_fused_dense_chain_length2_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length2_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length3_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length3_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length4_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length4_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length5_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length5_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length6_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length6_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length7_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length7_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length8_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length8_f32_vec4_no_tail_spirv)},
        }};
        static const std::array<spirv_blob, fixed_chain_variant_count> vector_restricted{{
            {eltwise_fused_dense_chain_length2_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length2_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length3_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length3_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length4_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length4_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length5_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length5_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length6_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length6_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length7_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length7_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length8_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length8_f32_vec4_no_tail_restrict_spirv)},
        }};
        if (kind == kernel_kind::fused_dense_chain_f32_vec4_no_tail) {
            select_fused_chain_spirv(vector_alias_safe, vector_restricted);
        } else {
            select_fused_chain_spirv(scalar_alias_safe, scalar_restricted);
        }
    } else if (kind == kernel_kind::broadcast_vector) {
        spirv = eltwise_broadcast_vector_spirv;
        spirv_size = sizeof(eltwise_broadcast_vector_spirv);
    } else if (kind == kernel_kind::broadcast_fast_scalar) {
        spirv = eltwise_broadcast_fast_spirv;
        spirv_size = sizeof(eltwise_broadcast_fast_spirv);
    } else if (kind == kernel_kind::broadcast_f32_eq) {
        spirv = eltwise_broadcast_f32_eq_spirv;
        spirv_size = sizeof(eltwise_broadcast_f32_eq_spirv);
    } else if (kind == kernel_kind::broadcast_fast_f32_eq) {
        spirv = eltwise_broadcast_fast_f32_eq_spirv;
        spirv_size = sizeof(eltwise_broadcast_fast_f32_eq_spirv);
    } else if (kind == kernel_kind::broadcast_fast_vector) {
        spirv = eltwise_broadcast_fast_vector_spirv;
        spirv_size = sizeof(eltwise_broadcast_fast_vector_spirv);
    } else if (kind == kernel_kind::unary) {
        spirv = eltwise_unary_spirv;
        spirv_size = sizeof(eltwise_unary_spirv);
    } else if (kind == kernel_kind::scalar_constant) {
        spirv = eltwise_scalar_constant_spirv;
        spirv_size = sizeof(eltwise_scalar_constant_spirv);
    } else {
        spirv = eltwise_spirv;
        spirv_size = sizeof(eltwise_spirv);
    }
    source->str.assign(reinterpret_cast<const char*>(spirv), spirv_size);
    source->entry_point = "main";
    source->batch_compilation = false;
    source->language = kernel_language::SPIRV;
    return source;
}

std::vector<std::shared_ptr<kernel_string>> make_kernel_sources(kernel_kind kind, uint32_t fused_chain_length) {
    std::vector<std::shared_ptr<kernel_string>> sources{make_kernel_source(kind, false, fused_chain_length)};
    if (supports_restricted_output(kind)) {
        sources.push_back(make_kernel_source(kind, true, fused_chain_length));
    }
    return sources;
}

bool output_is_disjoint_from_reads(const memory::cptr& output, const std::vector<memory::cptr>& reads) {
    const auto* output_buffer = output == nullptr ? nullptr : dynamic_cast<const vulkan_buffer*>(output.get());
    if (output_buffer == nullptr) {
        return false;
    }

    const auto output_begin = output_buffer->get_offset();
    const auto output_end = output_begin + output_buffer->size();
    for (const auto& read : reads) {
        const auto* read_buffer = read == nullptr ? nullptr : dynamic_cast<const vulkan_buffer*>(read.get());
        if (read_buffer == nullptr) {
            return false;
        }
        if (output_buffer->get_allocation() != read_buffer->get_allocation()) {
            continue;
        }
        const auto read_begin = read_buffer->get_offset();
        const auto read_end = read_begin + read_buffer->size();
        if (std::max(output_begin, read_begin) < std::min(output_end, read_end)) {
            return false;
        }
    }
    return true;
}

struct eltwise_impl : typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::eltwise_impl)

    eltwise_impl() : eltwise_impl(kernel_kind::broadcast_scalar, std::nullopt, 0, 0) {}

    eltwise_impl(kernel_kind kind,
                 std::optional<scalar_constant> scalar,
                 uint32_t elements_per_invocation,
                 uint32_t fused_chain_length)
        : parent("vulkan_eltwise"),
          _kernel_sources(make_kernel_sources(kind, fused_chain_length)),
          _kernel_kind(kind),
          _scalar_constant(std::move(scalar)),
          _elements_per_invocation(elements_per_invocation),
          _fused_chain_length(fused_chain_length),
          _local_size_tuning(std::make_shared<local_size_tuning_state>()) {}

    std::unique_ptr<primitive_impl> clone() const override {
        auto result = std::make_unique<eltwise_impl>(*this);
        result->_metadata_initialized = false;
        result->_local_size_tuning = std::make_shared<local_size_tuning_state>();
        return result;
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    void save(BinaryOutputBuffer& buffer) const override {
        parent::save(buffer);
        buffer << make_data(&_kernel_kind, sizeof(_kernel_kind));
        buffer << make_data(&_elements_per_invocation, sizeof(_elements_per_invocation));
        buffer << make_data(&_fused_chain_length, sizeof(_fused_chain_length));
        const bool has_scalar_constant = _scalar_constant.has_value();
        buffer << make_data(&has_scalar_constant, sizeof(has_scalar_constant));
        if (has_scalar_constant) {
            buffer << make_data(&_scalar_constant->input_index, sizeof(_scalar_constant->input_index));
            buffer << make_data(_scalar_constant->bits.data(), sizeof(_scalar_constant->bits));
        }
    }

    void load(BinaryInputBuffer& buffer) override {
        parent::load(buffer);
        buffer >> make_data(&_kernel_kind, sizeof(_kernel_kind));
        buffer >> make_data(&_elements_per_invocation, sizeof(_elements_per_invocation));
        buffer >> make_data(&_fused_chain_length, sizeof(_fused_chain_length));
        bool has_scalar_constant = false;
        buffer >> make_data(&has_scalar_constant, sizeof(has_scalar_constant));
        if (has_scalar_constant) {
            _scalar_constant.emplace();
            buffer >> make_data(&_scalar_constant->input_index, sizeof(_scalar_constant->input_index));
            buffer >> make_data(_scalar_constant->bits.data(), sizeof(_scalar_constant->bits));
        } else {
            _scalar_constant.reset();
        }
        _kernel_sources = make_kernel_sources(_kernel_kind, _fused_chain_length);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        if (uses_dense_push_constants(_kernel_kind)) {
            return {};
        }
        const auto buffer_words = is_fused_post_op_kernel(_kernel_kind)
                                      ? post_op_metadata_words
                                      : (is_fused_broadcast_kernel(_kernel_kind)
                                             ? fused_broadcast_metadata_words
                                             : (is_fused_chain_kernel(_kernel_kind)
                                                    ? fused_chain_metadata_words
                                                    : regular_metadata_words));
        return {BufferDescriptor(buffer_words, ov::element::u32, true, false)};
    }

    void init_kernels(const kernels_cache& cache, const kernel_impl_params& params) override {
        _kernels = cache.get_kernels(params);
        OPENVINO_ASSERT(_kernels.size() == _kernel_sources.size(), "[GPU][Vulkan] Eltwise compiled an unexpected number of SPIR-V kernels");
    }

    void init_by_cached_kernels(const kernels_cache& cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();
        for (const auto& id : cached_kernel_ids) {
            _kernels.push_back(cache.get_kernel_from_cached_kernels(id));
        }
        OPENVINO_ASSERT(_kernels.size() == 1 || (supports_restricted_output(_kernel_kind) && _kernels.size() == 2),
                        "[GPU][Vulkan] Cached Eltwise expects an alias-safe kernel and an optional restricted kernel");
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) override {
        return cache.get_cached_kernel_ids(_kernels);
    }

    std::vector<std::shared_ptr<kernel_string>> get_kernels_source() override {
        return _kernel_sources;
    }

    void reset_kernels_source() override {
        _kernel_sources.clear();
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    std::vector<size_t> get_in_place_input_indices() const override {
        return supports_restricted_output(_kernel_kind) ? std::vector<size_t>{0, 1} : std::vector<size_t>{};
    }

    void set_kernels(kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "[GPU][Vulkan] Eltwise expects one compiled kernel set");
        const auto& entries = kernels.begin()->second;
        OPENVINO_ASSERT(entries.size() == _kernel_sources.size(), "[GPU][Vulkan] Eltwise compiled an unexpected number of kernels");
        _kernels.resize(entries.size());
        for (const auto& entry : entries) {
            _kernels.at(entry.second) = entry.first;
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OPENVINO_ASSERT((_kernels.size() == 1 || _kernels.size() == 2) && _kernels.front() != nullptr, "[GPU][Vulkan] Eltwise kernel was not initialized");
        const auto desc = instance.get_typed_desc<eltwise>();
        const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
        const auto fused = get_supported_fused_eltwise_chain(instance.get_node());
        const auto post_op = get_supported_fused_post_op(instance.get_node());
        const bool use_fused_kernel = is_fused_kernel(_kernel_kind);
        const bool use_fused_post_op_kernel = is_fused_post_op_kernel(_kernel_kind);
        const bool use_fused_dense_kernel = is_fused_dense_kernel(_kernel_kind);
        const bool use_fused_broadcast_kernel = is_fused_broadcast_kernel(_kernel_kind);
        const bool use_single_fused_kernel = is_single_fused_kernel(_kernel_kind);
        const bool use_fused_chain_kernel = is_fused_chain_kernel(_kernel_kind);
        const bool use_dense_push_constants = uses_dense_push_constants(_kernel_kind);
        OPENVINO_ASSERT(use_fused_kernel == fused.has_value(),
                        "[GPU][Vulkan] Fused Eltwise kernel and graph metadata are inconsistent");
        OPENVINO_ASSERT(use_fused_post_op_kernel == post_op.has_value(),
                        "[GPU][Vulkan] Fused Eltwise post-op kernel and graph metadata are inconsistent");
        OPENVINO_ASSERT(!use_single_fused_kernel || fused->size() == 1,
                        "[GPU][Vulkan] Single-stage fused Eltwise kernel received a longer chain");
        OPENVINO_ASSERT(!use_fused_broadcast_kernel || fused->front().broadcast_input,
                        "[GPU][Vulkan] Fused broadcast Eltwise kernel received a dense external input");
        OPENVINO_ASSERT(!use_fused_chain_kernel ||
                            (fused->size() >= min_multi_stage_fused_chain_length && fused->size() <= max_fused_chain_length &&
                             fused->size() == _fused_chain_length),
                        "[GPU][Vulkan] Fused Eltwise chain kernel received an unsupported chain length");
        OPENVINO_ASSERT(use_fused_chain_kernel || _fused_chain_length == 0,
                        "[GPU][Vulkan] Non-chain Eltwise kernel received fixed chain metadata");
        OPENVINO_ASSERT(instance.inputs_memory_count() == base_input_count &&
                            instance.get_fused_mem_count() == (fused.has_value() ? fused->size() : 0U),
                        "[GPU][Vulkan] Eltwise received an unexpected regular or fused input count");
        OPENVINO_ASSERT(instance.get_intermediates_memories().size() == (use_dense_push_constants ? 0U : 1U),
                        "[GPU][Vulkan] Eltwise metadata resources do not match the selected kernel ABI");

        auto& stream = instance.get_network().get_stream();
        const auto metadata = make_metadata(instance, _scalar_constant, fused, post_op);
        memory::ptr metadata_memory;
        if (!use_dense_push_constants) {
            metadata_memory = instance.get_intermediates_memories().front();
            if (!_metadata_initialized || metadata != _cached_metadata) {
                metadata_memory->copy_from(stream, metadata.data(), true);
                _cached_metadata = metadata;
                _metadata_initialized = true;
            }
        }

        kernel_arguments_desc descriptor;
        descriptor.layerID = instance.id();
        const auto element_count = metadata[shader_abi::index(shader_abi::metadata_field::element_count)];
        const auto& input0_layout = instance.get_input_layout(0);
        const auto& input1_layout = base_input_count == 1 ? input0_layout : instance.get_input_layout(1);
        const auto& output_layout = instance.get_output_layout(0);
        const bool use_dense_kernel = is_plain_dense_kernel(_kernel_kind) || use_fused_kernel ||
                                      use_fused_post_op_kernel;
        const auto* fused_input_layout = use_fused_kernel
                                             ? &instance.get_input_layout(fused->front().external_dependency_index)
                                             : nullptr;
        const auto selected_dense_vector_width = get_dense_vector_width(_kernel_kind);
        const bool use_broadcast_vector_kernel = is_broadcast_vector_kernel(_kernel_kind);
        const bool use_fast_broadcast_kernel = is_fast_broadcast_kernel(_kernel_kind);
        const bool use_unary_kernel = _kernel_kind == kernel_kind::unary;
        const bool use_scalar_constant_kernel = _kernel_kind == kernel_kind::scalar_constant;
        OPENVINO_ASSERT(use_scalar_constant_kernel == _scalar_constant.has_value(),
                        "[GPU][Vulkan] Scalar Eltwise kernel and constant metadata are inconsistent");
        OPENVINO_ASSERT(!is_plain_dense_kernel(_kernel_kind) ||
                            (is_packed_dense_kernel(_kernel_kind) ? can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, nullptr)
                                                                  : can_use_dense_kernel(input0_layout, input1_layout, output_layout)),
                        "[GPU][Vulkan] Dense Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_fused_dense_kernel ||
                            (is_packed_dense_kernel(_kernel_kind)
                                 ? can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, fused_input_layout)
                                 : std::all_of(fused->begin(), fused->end(), [&](const fused_eltwise_info& fused_stage) {
                                       return can_use_fused_dense_kernel(input0_layout,
                                                                         input1_layout,
                                                                         instance.get_input_layout(fused_stage.external_dependency_index),
                                                                         output_layout);
                                   })),
                        "[GPU][Vulkan] Fused dense Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_fused_broadcast_kernel ||
                            (fused->size() == 1 && fused->front().broadcast_input &&
                             has_dense_storage(*fused_input_layout) &&
                             is_numpy_broadcast_compatible(*fused_input_layout, output_layout)),
                        "[GPU][Vulkan] Fused broadcast Eltwise runtime layouts no longer satisfy the compiled "
                        "kernel contract");
        OPENVINO_ASSERT(!use_fused_post_op_kernel ||
                            can_use_fused_post_op_kernel(input0_layout,
                                                         input1_layout,
                                                         *post_op,
                                                         output_layout),
                        "[GPU][Vulkan] Fused Eltwise post-op runtime layouts no longer satisfy the compiled "
                        "kernel contract");
        OPENVINO_ASSERT(selected_dense_vector_width == dense_vector_width::scalar ||
                            can_use_f32_dense_vector_width(input0_layout, input1_layout, output_layout, fused_input_layout, selected_dense_vector_width),
                        "[GPU][Vulkan] F32 vector Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_broadcast_vector_kernel || can_use_broadcast_vector_kernel(output_layout),
                        "[GPU][Vulkan] Vector broadcast Eltwise runtime layout no longer satisfies the compiled kernel contract");
        auto elements_per_invocation = _elements_per_invocation;
        if (elements_per_invocation == 0) {
            std::vector<eltwise_mode> fused_modes;
            if (fused.has_value()) {
                fused_modes.reserve(fused->size());
                for (const auto& fused_stage : *fused) {
                    fused_modes.push_back(fused_stage.descriptor->typed_desc<eltwise>()->mode);
                }
            }
            elements_per_invocation = select_generic_elements_per_invocation(_kernel_kind,
                                                                             input0_layout,
                                                                             input1_layout,
                                                                             output_layout,
                                                                             fused_input_layout,
                                                                             desc->mode,
                                                                             fused_modes,
                                                                             instance.get_network().get_engine().get_device_info());
        }
        OPENVINO_ASSERT(!use_fast_broadcast_kernel || should_use_fast_broadcast_kernel(input0_layout,
                                                                                       input1_layout,
                                                                                       output_layout,
                                                                                       instance.get_network().get_engine().get_device_info(),
                                                                                       elements_per_invocation),
                        "[GPU][Vulkan] Fast broadcast Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        const auto invocation_count = (element_count + elements_per_invocation - 1) / elements_per_invocation;
        const auto& device_info = instance.get_network().get_engine().get_device_info();
        local_size_tuning_action local_size_action;
        std::unique_lock<std::mutex> local_size_tuning_lock;
        if (_elements_per_invocation != 0) {
            const auto cached_local_size = _local_size_tuning->cached_selected.load(std::memory_order_acquire);
            if (cached_local_size != 0) {
                local_size_action.local_size = cached_local_size;
            } else {
                local_size_tuning_lock = std::unique_lock<std::mutex>(_local_size_tuning->mutex);
                local_size_action = next_local_size_action(*_local_size_tuning, invocation_count, device_info, is_no_tail_kernel(_kernel_kind));
                if (!local_size_action.prewarm && !local_size_action.measure) {
                    local_size_tuning_lock.unlock();
                }
            }
        } else {
            local_size_action.local_size = select_local_work_group_size(invocation_count, device_info.max_work_group_size);
        }
        descriptor.workGroups.global = {invocation_count, 1, 1};
        descriptor.workGroups.local = {local_size_action.local_size, 1, 1};
        descriptor.specialize_local_size_x = true;
        descriptor.specialization_constants = {
            {shader_abi::index(shader_abi::specialization_id::mode), metadata[shader_abi::index(shader_abi::metadata_field::mode)]},
            {shader_abi::index(shader_abi::specialization_id::input0_type), metadata[shader_abi::index(shader_abi::metadata_field::input0_type)]},
            {shader_abi::index(shader_abi::specialization_id::input1_type), metadata[shader_abi::index(shader_abi::metadata_field::input1_type)]},
            {shader_abi::index(shader_abi::specialization_id::output_type), metadata[shader_abi::index(shader_abi::metadata_field::output_type)]},
            {shader_abi::index(shader_abi::specialization_id::storage_flags), metadata[shader_abi::index(shader_abi::metadata_field::flags)]},
        };
        if (use_dense_kernel || use_broadcast_vector_kernel || use_scalar_constant_kernel) {
            descriptor.specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::elements_per_invocation), elements_per_invocation});
        }
        if (use_fast_broadcast_kernel) {
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_rank), metadata[shader_abi::index(shader_abi::metadata_field::rank)]});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_input0_axes), active_broadcast_axes(metadata, shader_abi::tensor_index::input0)});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_input1_axes), active_broadcast_axes(metadata, shader_abi::tensor_index::input1)});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_output_axes), active_broadcast_axes(metadata, shader_abi::tensor_index::output)});
        }
        if (use_single_fused_kernel) {
            descriptor.specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_mode),
                                                           metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::mode)]});
            descriptor.specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_input_type),
                                                           metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_type)]});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::fused_input_position),
                 metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_position)]});
        } else if (use_fused_chain_kernel) {
            for (size_t stage = 0; stage < fused->size(); ++stage) {
                const auto stage_index = checked_u32(stage, "fused specialization stage");
                const auto stage_metadata_base = fused_metadata_base + stage * fused_metadata_words;
                descriptor.specialization_constants.push_back(
                    {shader_abi::index(shader_abi::specialization_id::fused_chain_mode_base) + stage_index,
                     metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::mode)]});
                descriptor.specialization_constants.push_back(
                    {shader_abi::index(shader_abi::specialization_id::fused_chain_input_type_base) + stage_index,
                     metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_type)]});
                descriptor.specialization_constants.push_back(
                    {shader_abi::index(shader_abi::specialization_id::fused_chain_input_position_base) + stage_index,
                     metadata[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_position)]});
            }
        }
        if (use_fused_post_op_kernel) {
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::base_output_type),
                 shader_abi::value(scalar_type_code(post_op->descriptor->input_layout.data_type))});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::post_op_kind),
                 shader_abi::value(post_op->kind)});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::post_activation),
                 shader_abi::value(post_op->activation)});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::post_quantize_flags),
                 post_op->quantize_flags});
        }
        if (use_dense_push_constants) {
            descriptor.scalars = make_dense_push_constants(metadata, is_single_fused_dense_kernel(_kernel_kind));
        }
        descriptor.arguments = {{argument_desc::Types::INPUT, 0}};
        if (use_scalar_constant_kernel) {
            descriptor.arguments.front().index = _scalar_constant->input_index == shader_abi::tensor_index::input0
                                                     ? shader_abi::index(shader_abi::tensor_index::input1)
                                                     : shader_abi::index(shader_abi::tensor_index::input0);
        } else if (!use_unary_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::INPUT, 1});
        }
        if (use_single_fused_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
        } else if (use_fused_chain_kernel) {
            for (size_t stage = 0; stage < max_fused_chain_length; ++stage) {
                descriptor.arguments.push_back(
                    {argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE, static_cast<uint32_t>(std::min(stage, fused->size() - 1))});
            }
        }
        descriptor.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        if (!use_dense_push_constants) {
            descriptor.arguments.push_back({argument_desc::Types::INTERNAL_BUFFER, 0});
        }
        if (!use_dense_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        }

        kernel_arguments_data arguments;
        for (size_t input_index = 0; input_index < base_input_count; ++input_index) {
            arguments.inputs.push_back(instance.input_memory_ptr(input_index));
        }
        if (use_fused_kernel) {
            for (size_t stage = 0; stage < fused->size(); ++stage) {
                arguments.fused_op_inputs.push_back(instance.fused_memory(stage));
            }
        }
        arguments.outputs = {instance.output_memory_ptr(0)};
        if (!use_dense_push_constants) {
            arguments.intermediates = {metadata_memory};
        }
        std::vector<memory::cptr> read_memories;
        if (use_scalar_constant_kernel) {
            const auto tensor_input_index = _scalar_constant->input_index == shader_abi::tensor_index::input0
                                                ? shader_abi::index(shader_abi::tensor_index::input1)
                                                : shader_abi::index(shader_abi::tensor_index::input0);
            read_memories.push_back(arguments.inputs.at(tensor_input_index));
        } else {
            read_memories.insert(read_memories.end(), arguments.inputs.begin(), arguments.inputs.end());
        }
        read_memories.insert(read_memories.end(), arguments.fused_op_inputs.begin(), arguments.fused_op_inputs.end());
        read_memories.insert(read_memories.end(), arguments.intermediates.begin(), arguments.intermediates.end());
        const auto use_restricted_kernel = _kernels.size() == 2 && output_is_disjoint_from_reads(arguments.outputs.front(), read_memories);
        auto& selected_kernel = *_kernels.at(use_restricted_kernel ? 1 : 0);
        if (local_size_action.prewarm) {
            for (const auto candidate : _local_size_tuning->candidates) {
                descriptor.workGroups.local[0] = candidate;
                stream.set_arguments(selected_kernel, descriptor, arguments);
            }
            _local_size_tuning->prewarmed = true;
            local_size_tuning_lock.unlock();
            descriptor.workGroups.local[0] = _local_size_tuning->fallback;
        }
        if (local_size_action.measure) {
            stream.finish();
            const auto start = std::chrono::steady_clock::now();
            auto measured_event = stream.enqueue_kernel(selected_kernel, descriptor, arguments, events, true);
            OPENVINO_ASSERT(measured_event != nullptr, "[GPU][Vulkan] Eltwise local-size measurement did not produce a completion event");
            measured_event->wait();
            const auto elapsed = std::chrono::steady_clock::now() - start;
            _local_size_tuning->samples[local_size_action.candidate_index].push_back(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
            local_size_tuning_lock.unlock();
            return measured_event;
        }
        return stream.enqueue_kernel(selected_kernel, descriptor, arguments, events, instance.needs_completion_event());
    }

private:
    std::vector<std::shared_ptr<kernel_string>> _kernel_sources;
    kernel_kind _kernel_kind = kernel_kind::broadcast_scalar;
    std::optional<scalar_constant> _scalar_constant;
    uint32_t _elements_per_invocation = 0;
    uint32_t _fused_chain_length = 0;
    std::shared_ptr<local_size_tuning_state> _local_size_tuning;
    std::vector<kernel::ptr> _kernels;
    std::array<uint32_t, metadata_words> _cached_metadata{};
    bool _metadata_initialized = false;
};

}  // namespace

bool EltwiseImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }

    const auto& desc = node.as<eltwise>().get_primitive();
    const auto expected_inputs = is_unary_mode(desc->mode) ? 1U : 2U;
    const auto fused = get_supported_fused_eltwise_chain(node);
    const auto post_op = get_supported_fused_post_op(node);
    if (node.has_fused_primitives() && !fused.has_value() && !post_op.has_value()) {
        return false;
    }
    const bool coefficients_supported = desc->coefficients.empty() || (desc->mode == eltwise_mode::sum && desc->coefficients.size() == expected_inputs) ||
                                        (desc->mode == eltwise_mode::is_inf && desc->coefficients.size() == 2);
    if (!is_supported_mode(desc->mode) ||
        node.get_dependencies().size() != expected_inputs + (fused.has_value() ? fused->size() : 0U) ||
        !coefficients_supported ||
        !desc->stride.empty() ||
        (desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NUMPY && desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NONE)) {
        return false;
    }

    for (size_t index = 0; index < expected_inputs; ++index) {
        const auto& input_layout = node.get_input_layout(index);
        if (!is_supported_data_type(input_layout.data_type) || !is_supported_format(input_layout.format.value) || input_layout.get_rank() > max_rank) {
            return false;
        }
        if (is_bitwise_mode(desc->mode) && !is_integer_data_type(input_layout.data_type)) {
            return false;
        }
        if ((desc->mode == eltwise_mode::atan2 || is_unary_mode(desc->mode)) && !data_type_traits::is_floating_point(input_layout.data_type)) {
            return false;
        }
    }
    const auto& output_layout = node.get_output_layout(0);
    return is_supported_data_type(output_layout.data_type) && is_supported_format(output_layout.format.value) && output_layout.get_rank() <= max_rank;
}

std::unique_ptr<primitive_impl> EltwiseImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    const auto* engine = dynamic_cast<const vulkan_engine*>(&node.get_program().get_engine());
    OPENVINO_ASSERT(engine != nullptr, "[GPU][Vulkan] Eltwise implementation requires a Vulkan engine");
    const auto max_push_constants_size = engine->get_max_push_constants_size();
    const auto& device_info = engine->get_device_info();
    const auto input_count = params.input_layouts.size();
    const auto fused = get_supported_fused_eltwise_chain(node);
    const auto post_op = get_supported_fused_post_op(node);
    const auto base_input_count = is_unary_mode(node.as<eltwise>().get_primitive()->mode) ? 1U : 2U;
    OPENVINO_ASSERT(input_count == base_input_count + (fused.has_value() ? fused->size() : 0U),
                    "[GPU][Vulkan] Eltwise implementation creation received an unexpected input count");
    const auto& input0_layout = params.get_input_layout(0);
    const auto& input1_layout = input_count == 1 ? input0_layout : params.get_input_layout(1);
    kernel_kind kind = kernel_kind::broadcast_scalar;
    std::optional<scalar_constant> scalar;
    const auto fused_chain_length = fused.has_value() && fused->size() >= min_multi_stage_fused_chain_length
                                        ? checked_u32(fused->size(), "fused chain length")
                                        : 0U;
    if (post_op.has_value()) {
        kind = kernel_kind::fused_post_op;
    } else if (fused.has_value()) {
        kind = fused->size() == 1
                   ? (fused->front().broadcast_input ? kernel_kind::fused_broadcast : kernel_kind::fused_dense)
                   : kernel_kind::fused_dense_chain;
        if (fused_chain_length != 0) {
            const auto& fused_input_layout = params.get_input_layout(fused->front().external_dependency_index);
            const auto vector_width =
                select_f32_dense_vector_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
            if (vector_width == dense_vector_width::vec4 && can_use_f32_no_tail_kernel(params.get_output_layout(0), vector_width, device_info)) {
                kind = kernel_kind::fused_dense_chain_f32_vec4_no_tail;
            }
        } else if (!fused->front().broadcast_input && max_push_constants_size >= fused_dense_push_constant_bytes) {
            const auto& fused_input_layout = params.get_input_layout(fused->front().external_dependency_index);
            const auto packed_width = select_packed_dense_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
            if (packed_width != packed_dense_width::none) {
                kind = select_packed_dense_kernel_kind(packed_width, params.get_output_layout(0).data_type, true);
            } else {
                const auto vector_width =
                    select_f32_dense_vector_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
                const auto no_tail =
                    vector_width != dense_vector_width::scalar && can_use_f32_no_tail_kernel(params.get_output_layout(0), vector_width, device_info);
                kind = vector_width == dense_vector_width::scalar ? kernel_kind::fused_dense_push_constants
                                                                  : select_f32_vector_kernel_kind(vector_width, true, no_tail);
            }
        }
    } else if (is_unary_mode(node.as<eltwise>().get_primitive()->mode)) {
        kind = kernel_kind::unary;
    } else if (!params.is_dynamic()) {
        const auto& output_layout = params.get_output_layout(0);
        scalar = get_scalar_constant(node);
        if (scalar.has_value() && output_layout.count() >= broadcast_vector_min_elements) {
            kind = kernel_kind::scalar_constant;
        } else if (can_use_dense_kernel(input0_layout, input1_layout, output_layout) ||
                   can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, nullptr)) {
            kind = can_use_dense_kernel(input0_layout, input1_layout, output_layout) ? kernel_kind::dense : kernel_kind::broadcast_scalar;
            if (max_push_constants_size >= dense_push_constant_bytes) {
                const auto packed_width = select_packed_dense_width(input0_layout, input1_layout, output_layout, nullptr, device_info);
                if (packed_width != packed_dense_width::none) {
                    kind = select_packed_dense_kernel_kind(packed_width, output_layout.data_type, false);
                } else if (can_use_dense_kernel(input0_layout, input1_layout, output_layout)) {
                    const auto vector_width = select_f32_dense_vector_width(input0_layout, input1_layout, output_layout, nullptr, device_info);
                    const auto no_tail = vector_width != dense_vector_width::scalar && can_use_f32_no_tail_kernel(output_layout, vector_width, device_info);
                    kind = vector_width == dense_vector_width::scalar ? kernel_kind::dense_push_constants
                                                                      : select_f32_vector_kernel_kind(vector_width, false, no_tail);
                } else {
                    kind = kernel_kind::broadcast_scalar;
                }
            }
        } else {
            const bool vector_kernel = can_use_broadcast_vector_kernel(output_layout);
            const auto provisional_kind = vector_kernel ? kernel_kind::broadcast_vector : kernel_kind::broadcast_scalar;
            const auto elements_per_invocation = select_generic_elements_per_invocation(provisional_kind,
                                                                                        input0_layout,
                                                                                        input1_layout,
                                                                                        output_layout,
                                                                                        nullptr,
                                                                                        node.as<eltwise>().get_primitive()->mode,
                                                                                        {},
                                                                                        device_info);
            const bool fast_kernel = should_use_fast_broadcast_kernel(input0_layout, input1_layout, output_layout, device_info, elements_per_invocation);
            kind = vector_kernel ? (fast_kernel ? kernel_kind::broadcast_fast_vector : kernel_kind::broadcast_vector)
                                 : (fast_kernel ? kernel_kind::broadcast_fast_scalar : kernel_kind::broadcast_scalar);
        }
        if (kind != kernel_kind::scalar_constant) {
            scalar.reset();
        }
        kind = select_pre_specialized_kernel_kind(kind, node.as<eltwise>().get_primitive()->mode, input0_layout, input1_layout, params.get_output_layout(0));
    }
    uint32_t elements_per_invocation = 0;
    if (!params.is_dynamic()) {
        const auto* fused_input_layout = fused.has_value() ? &params.get_input_layout(fused->front().external_dependency_index) : nullptr;
        std::vector<eltwise_mode> fused_modes;
        if (fused.has_value()) {
            fused_modes.reserve(fused->size());
            for (const auto& fused_stage : *fused) {
                fused_modes.push_back(fused_stage.descriptor->typed_desc<eltwise>()->mode);
            }
        }
        elements_per_invocation = select_generic_elements_per_invocation(kind,
                                                                         input0_layout,
                                                                         input1_layout,
                                                                         params.get_output_layout(0),
                                                                         fused_input_layout,
                                                                         node.as<eltwise>().get_primitive()->mode,
                                                                         fused_modes,
                                                                         device_info);
    }
    return std::make_unique<eltwise_impl>(kind, std::move(scalar), elements_per_invocation, fused_chain_length);
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::eltwise_impl)
