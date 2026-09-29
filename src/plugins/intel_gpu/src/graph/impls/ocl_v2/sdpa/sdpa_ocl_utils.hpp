// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "ocl_v2/utils/jitter.hpp"

// Host helpers shared by SDPAOclGenerator and SDPAOclDecodeGenerator. The derivations behind them are in
// src/plugins/intel_gpu/docs/sdpa_ocl.md.
namespace ov::intel_gpu::ocl::sdpa_ocl_utils {

// Debug and tuning switches, read when a kernel is generated.
inline bool env_set(const char* name) {
    return std::getenv(name) != nullptr;
}

// atoi() of the variable, or `fallback` when it is not set.
inline int env_int(const char* name, int fallback) {
    if (const char* env = std::getenv(name)) {
        return std::atoi(env);
    }
    return fallback;
}

// Whether the variable starts with '1', or `unset` when it is not set.
inline bool env_on(const char* name, bool unset = false) {
    const char* env = std::getenv(name);
    return env == nullptr ? unset : env[0] == '1';
}

inline size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

// 2D block IO surface rules: width >= 64 B and a multiple of 4, pitch >= 64 B and a multiple of 16, and a
// 64 B aligned base. Callers pass the per-head row, of which the pitch and the base offset are integer
// multiples, so each rule reduces to a test on row_bytes ("Block2d rules" in the docs).

// row_bytes % 64 == 0 makes every base 64 B aligned as well, so no fixup is needed.
inline bool block2d_surface_ok(size_t row_bytes) {
    return row_bytes >= 64 && (row_bytes % 64) == 0;
}

// Width and pitch only: the base can then be off by a multiple of 16 bytes.
inline bool block2d_width_pitch_ok(size_t row_bytes) {
    return row_bytes >= 64 && (row_bytes % 16) == 0;
}

// A paged-attention cache page. Its base is a whole number of pages, and for every cache layout the pitch
// rule already makes the page stride a multiple of 64, so width and pitch are all that is left.
inline bool block2d_page_ok(size_t row_bytes) {
    return block2d_width_pitch_ok(row_bytes);
}

// Whether `ch` carries no static or dynamic padding. Indexed the way ocl_v2/utils/jitter.cpp reads the
// padding arrays; an axis the format does not have cannot be padded.
inline bool axis_unpadded(const cldnn::layout& l, ChannelName ch) {
    const auto rank = l.get_partial_shape().size();
    const int idx = get_channel_index(ch, rank, cldnn::format::is_weights_format(l.format), cldnn::format::is_grouped(l.format));
    if (idx < 0 || idx >= static_cast<int>(rank))
        return true;
    const auto& pad = l.data_padding;
    return pad._lower_size.at(idx) == 0 && pad._upper_size.at(idx) == 0 && !pad._dynamic_dims_mask[idx];
}

// A paged-attention Q/K/V is a rank-2 token matrix [tokens, heads * head_size]. The head dimension lives inside
// FEATURE, so feature padding moves both the base (by the padding before) and the token stride of every head by
// an arbitrary amount: the argument "the pitch and the base are integer multiples of the row" that holds for a
// rank-4 [b, h, s, d] tensor does not apply, and axis_unpadded(X) is vacuously true there (the format has no X).
inline bool is_token_matrix(const cldnn::layout& l) {
    return l.get_partial_shape().size() == 2;
}

inline bool layout_unpadded(const cldnn::layout& l) {
    return !l.data_padding && !l.data_padding.is_dynamic();
}

// Bytes before the first head of a token, and the token stride, of a token matrix whose padding is static. False
// when they cannot be known at compile time: a dynamic padding (the sizes arrive with the shape) or a feature
// dimension that is not static.
inline bool token_matrix_bytes(const cldnn::layout& l, int64_t& first_head, int64_t& token_stride) {
    const auto& pshape = l.get_partial_shape();
    if (!is_token_matrix(l) || l.data_padding.is_dynamic() || pshape[1].is_dynamic())
        return false;
    const int idx = get_channel_index(ChannelName::FEATURE, 2, cldnn::format::is_weights_format(l.format), cldnn::format::is_grouped(l.format));
    if (idx != 1)
        return false;
    const int64_t elt = static_cast<int64_t>(ov::element::Type(l.data_type).size());
    const auto& pad = l.data_padding;
    first_head = pad._lower_size.at(idx) * elt;
    token_stride = (pshape[1].get_length() + pad._lower_size.at(idx) + pad._upper_size.at(idx)) * elt;
    return true;
}

// A tensor whose base needs no repair: a 64 B aligned base and a pitch that is a multiple of 64 for every head of
// every token, given a 64 B aligned buffer (true for engine allocations and for in-place crop views of them, whose
// offset is folded into the padding). Unpadded, that is row_bytes % 64. A padded rank-4 tensor keeps it as long as its innermost axis is
// unpadded (every offset is then a whole number of rows). A padded token matrix must prove it from static padding,
// and a dynamic padding cannot be proven, so it never takes this tier.
inline bool block2d_layout_ok(const cldnn::layout& l, size_t row_bytes) {
    if (!block2d_surface_ok(row_bytes))
        return false;
    if (layout_unpadded(l))
        return true;
    if (is_token_matrix(l)) {
        int64_t first_head = 0;
        int64_t token_stride = 0;
        return token_matrix_bytes(l, first_head, token_stride) && first_head % 64 == 0 && token_stride % 64 == 0;
    }
    // X is the innermost axis only at rank 4 (at rank 3 it does not exist and the test would be vacuously true).
    return l.get_partial_shape().size() == 4 && cldnn::format::is_simple_data_format(l.format) && axis_unpadded(l, ChannelName::X);
}

// A tensor whose base the kernel repairs (BLOCK2D_KV_BASE_FIXUP), by rounding it down by up to 60 bytes (base & 63).
// The pitch must still be a multiple of 16 and the base a multiple of 4 (the widened surface stays a multiple of 4
// wide). A padded rank-4 tensor keeps that as long as its innermost axis is unpadded. A padded token matrix with
// static padding is checked, at 16 bytes for both the start of the first head and the stride (stricter than the
// device needs, like the 64 B of the strict tier). A dynamic one is taken on trust: the kernel reads the sizes at
// run time, and the tier assumes a token stride that is a multiple of 16 B and a first-head start that is a multiple
// of 4 B, as a crop view of a fused QKV tensor at head-size offsets has. A dynamic padding that breaks it reads
// wrong values ("Block2d rules" in the docs).
inline bool block2d_layout_fixup_ok(const cldnn::layout& l, size_t row_bytes) {
    if (!block2d_width_pitch_ok(row_bytes))
        return false;
    if (layout_unpadded(l))
        return true;
    if (is_token_matrix(l)) {
        if (l.data_padding.is_dynamic())
            return true;
        int64_t first_head = 0;
        int64_t token_stride = 0;
        return token_matrix_bytes(l, first_head, token_stride) && first_head % 16 == 0 && token_stride % 16 == 0;
    }
    return l.get_partial_shape().size() == 4 && cldnn::format::is_simple_data_format(l.format) && axis_unpadded(l, ChannelName::X);
}

// The configured kv-cache precision. A u4 cache is materialized as a u8 tensor (and an i4 one as i8), so
// the KEY_CACHE layout type cannot tell them apart.
inline ov::element::Type pa_kv_cache_precision(const cldnn::kernel_impl_params& params) {
    return params.get_program().get_config().get_kv_cache_precision();
}

// Data-row pitch of a u4 K page in bytes: exactly h / 2, not aligned, which is what makes the token-major
// page (16 * (h / 2) data + 4 * h comp bytes = 12 * h) fit the upstream d-major INT4 page.
inline size_t pa_u4_k_row_bytes(size_t k_head_size) {
    return k_head_size / 2;
}

// Data-row pitch of a u4 V page in bytes, aligned to the subgroup; the trailing comp slack absorbs it.
inline size_t pa_u4_v_row_bytes(size_t v_head_size, size_t subgroup_size) {
    return align_up(v_head_size / 2, subgroup_size);
}

// Whether a BY_CHANNEL K cache was materialized token-major, read from the physical shape (the decision is
// made once, in transformations_pipeline.cpp). The adjusted block size is the page's per-channel row
// count: block_size (block_size / 2 for int4) plus the comp bytes.
inline bool pa_k_by_channel_tm_layout(const cldnn::layout& key_cache, bool int4, size_t comp_bytes) {
    const size_t block = cldnn::paged_attention::block_size;
    return cldnn::paged_attention::k_by_channel_token_major_layout(key_cache.get_partial_shape(), (int4 ? block / 2 : block) + comp_bytes);
}

// A K page is ADJUSTED_K_HEAD_SIZE x ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE and a V page block_size x
// ADJUSTED_V_HEAD_SIZE. The comp region grows the factor it is indexed by (per token: the head axis, per
// channel: the block axis) and int4 halves the packed block axis. comp_bytes is 0 when uncompressed.
inline void add_pa_adjusted_jit(JitConstants& jit, size_t k_head_size, size_t v_row_elems, bool int4, bool key_by_channel, size_t comp_bytes) {
    const size_t block = cldnn::paged_attention::block_size;
    jit.make("ADJUSTED_K_HEAD_SIZE", k_head_size + (key_by_channel ? 0 : comp_bytes));
    jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", (int4 ? block / 2 : block) + (key_by_channel ? comp_bytes : 0));
    jit.make("ADJUSTED_V_HEAD_SIZE", v_row_elems + comp_bytes);
}

inline void add_sink_jit(JitConstants& jit, const cldnn::layout& sink) {
    jit.make("SINK_DATA_T", to_ocl_type(sink.data_type));
    jit.make("HAS_SINK_INPUT", 1);
}

// INPUT<i> describes the i-th of `input_ids` (the kernel's parameter order, not the op's), then OUTPUT.
inline void add_io_layouts_jit(JitConstants& jit, const cldnn::kernel_impl_params& params, std::initializer_list<size_t> input_ids) {
    size_t i = 0;
    for (const size_t id : input_ids) {
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i++), params.input_layouts[id], params.in_port_to_shape_info_offset.at(id)));
    }
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], params.out_port_to_shape_info_offset.at(0)));
}

}  // namespace ov::intel_gpu::ocl::sdpa_ocl_utils
