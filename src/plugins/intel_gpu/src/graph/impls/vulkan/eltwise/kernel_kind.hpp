// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "intel_gpu/runtime/utils.hpp"

namespace cldnn::vulkan::eltwise_detail {

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

inline bool is_f32_vector_kernel(kernel_kind kind) {
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

inline bool is_no_tail_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_chain_f32_vec4_no_tail});
}

inline bool is_packed_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_packed_8bit_push_constants,
                   kernel_kind::dense_packed_16bit_push_constants,
                   kernel_kind::fused_dense_packed_8bit_push_constants,
                   kernel_kind::fused_dense_packed_16bit_push_constants,
                   kernel_kind::dense_packed_f16_push_constants,
                   kernel_kind::fused_dense_packed_f16_push_constants});
}

inline bool uses_dense_push_constants(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_push_constants,
                   kernel_kind::fused_dense_push_constants,
                   kernel_kind::dense_f32_sum_push_constants,
                   kernel_kind::dense_f32_div_push_constants,
                   kernel_kind::dense_i64_sum_push_constants,
                   kernel_kind::dense_i64_div_push_constants}) ||
           (is_f32_vector_kernel(kind) && kind != kernel_kind::fused_dense_chain_f32_vec4_no_tail) || is_packed_dense_kernel(kind);
}

inline bool is_broadcast_vector_kernel(kernel_kind kind) {
    return one_of(kind, {kernel_kind::broadcast_vector, kernel_kind::broadcast_fast_vector});
}

inline bool is_fast_broadcast_kernel(kernel_kind kind) {
    return one_of(kind, {kernel_kind::broadcast_fast_scalar, kernel_kind::broadcast_fast_vector, kernel_kind::broadcast_fast_f32_eq});
}

inline bool is_plain_dense_kernel(kernel_kind kind) {
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

inline bool is_single_fused_dense_kernel(kernel_kind kind) {
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

inline bool is_fused_broadcast_kernel(kernel_kind kind) {
    return kind == kernel_kind::fused_broadcast;
}

inline bool is_fused_post_op_kernel(kernel_kind kind) {
    return kind == kernel_kind::fused_post_op;
}

inline bool is_single_fused_kernel(kernel_kind kind) {
    return is_single_fused_dense_kernel(kind) || is_fused_broadcast_kernel(kind);
}

inline bool is_fused_chain_kernel(kernel_kind kind) {
    return one_of(kind, {kernel_kind::fused_dense_chain, kernel_kind::fused_dense_chain_f32_vec4_no_tail});
}

inline bool is_fused_dense_kernel(kernel_kind kind) {
    return is_single_fused_dense_kernel(kind) || is_fused_chain_kernel(kind);
}

inline bool is_fused_kernel(kernel_kind kind) {
    return is_fused_dense_kernel(kind) || is_fused_broadcast_kernel(kind);
}

inline bool supports_restricted_output(kernel_kind kind) {
    return is_plain_dense_kernel(kind) || is_fused_dense_kernel(kind);
}

inline dense_vector_width get_dense_vector_width(kernel_kind kind) {
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

inline packed_dense_width get_packed_dense_width(kernel_kind kind) {
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

}  // namespace cldnn::vulkan::eltwise_detail
