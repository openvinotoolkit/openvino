// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/utils/paged_cache_manager.hpp"

namespace ov {
namespace reference {

namespace detail {

inline float read_scalar_as_f32(const void* ptr, const ov::element::Type& et) {
    OPENVINO_ASSERT(ptr != nullptr, "read_scalar_as_f32: null pointer");
    if (et == ov::element::f32) {
        return *static_cast<const float*>(ptr);
    }
    if (et == ov::element::f64) {
        return static_cast<float>(*static_cast<const double*>(ptr));
    }
    if (et == ov::element::f16) {
        return static_cast<float>(*static_cast<const ov::float16*>(ptr));
    }
    if (et == ov::element::bf16) {
        return static_cast<float>(*static_cast<const ov::bfloat16*>(ptr));
    }
    OPENVINO_THROW("PagedAttention reference: unsupported scalar type: ", et);
}

inline float read_at_as_f32(const void* base, const ov::element::Type& et, std::size_t idx) {
    OPENVINO_ASSERT(base != nullptr, "read_at_as_f32: null pointer");
    if (et == ov::element::f32) {
        return static_cast<const float*>(base)[idx];
    }
    if (et == ov::element::f64) {
        return static_cast<float>(static_cast<const double*>(base)[idx]);
    }
    if (et == ov::element::f16) {
        return static_cast<float>(static_cast<const ov::float16*>(base)[idx]);
    }
    if (et == ov::element::bf16) {
        return static_cast<float>(static_cast<const ov::bfloat16*>(base)[idx]);
    }
    OPENVINO_THROW("PagedAttention reference: unsupported array type: ", et);
}

inline void softmax_inplace(std::vector<float>& scores) {
    if (scores.empty()) {
        return;
    }
    const float m = *std::max_element(scores.begin(), scores.end());
    float sum = 0.f;
    for (float& s : scores) {
        s = std::exp(s - m);
        sum += s;
    }
    const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (float& s : scores) {
        s *= inv;
    }
}

inline void apply_rotary_inplace(std::vector<float>& vec,
                                 const void* trig_lut,
                                 const ov::element::Type& trig_et,
                                 std::size_t trig_row,
                                 std::size_t head_size) {
    if (trig_lut == nullptr || head_size < 2 || (head_size % 2) != 0) {
        return;
    }
    const std::size_t half = head_size / 2;
    const std::size_t row_base = trig_row * head_size;
    for (std::size_t i = 0; i < half; ++i) {
        const float x0 = vec[2 * i];
        const float x1 = vec[2 * i + 1];
        const float c = read_at_as_f32(trig_lut, trig_et, row_base + i);
        const float s = read_at_as_f32(trig_lut, trig_et, row_base + half + i);
        vec[2 * i] = x0 * c - x1 * s;
        vec[2 * i + 1] = x0 * s + x1 * c;
    }
}

inline std::vector<std::size_t> parse_subsequence_ranges(const int32_t* subseq_begins,
                                                         std::size_t subseq_count,
                                                         std::size_t seq_count,
                                                         std::size_t batch_tokens) {
    // Returns a vector of size (seq_count + 1) with begins and a final end
    std::vector<std::size_t> begins(seq_count + 1, 0);
    if (subseq_begins == nullptr || subseq_count == 0 || seq_count == 0) {
        begins[0] = 0;
        begins[seq_count] = batch_tokens;
        return begins;
    }

    const std::size_t usable = std::min(subseq_count, seq_count);
    for (std::size_t s = 0; s < usable; ++s) {
        const std::int32_t v = subseq_begins[s];
        begins[s] = (v < 0) ? 0 : static_cast<std::size_t>(v);
    }

    if (subseq_count >= seq_count + 1) {
        const std::int32_t vend = subseq_begins[seq_count];
        begins[seq_count] = (vend < 0) ? batch_tokens : static_cast<std::size_t>(vend);
    } else {
        begins[seq_count] = batch_tokens;
    }

    // Clamp and monotonic fix.
    begins[0] = std::min(begins[0], batch_tokens);
    for (std::size_t s = 1; s <= seq_count; ++s) {
        begins[s] = std::min(begins[s], batch_tokens);
        if (begins[s] < begins[s - 1]) {
            begins[s] = begins[s - 1];
        }
    }
    return begins;
}

}  // namespace detail

// Reference implementation of ov::op::PagedAttentionExtension
//
// This implementation currently does the following:
// - It stores KV into the provided PagedCacheManager (copied from init cache once per node)
// - It computes causal attention for the newly provided tokens (query/key/value inputs)
// - It supports GQA (num_heads != num_kv_heads)
// - It supports ALiBi (optional vector; broadcast if length == 1)
// - It supports basic RoPE re-rotation of cached blocks (rotated_block_indices + rotation_deltas + trig LUT)
//
// However, the implementation does not yet support advanced cache-eviction related inputs,
// And those are currently ignored by the reference kernel
// They remain in the function signature so the operator can be evaluated with the full input list
template <typename T>
void paged_attention(std::uintptr_t node_key,
                     ov::reference::paged_attention_cache::PagedCacheManager* cache_manager,
                     T* out,
                     T* out_scores,
                     T* out_aux,
                     const T* query,
                     const T* key,
                     const T* value,
                     const T* key_cache_init,
                     const T* value_cache_init,
                     const int32_t* past_lens,
                     const int32_t* subsequence_begins,
                     const int32_t* block_indices_init,
                     std::size_t block_indices_count,
                     const int32_t* block_indices_begins_init,
                     std::size_t block_indices_begins_count,
                     const void* scale,
                     const ov::element::Type& scale_et,
                     const int32_t* sliding_window,
                     const void* alibi_slopes,
                     const ov::element::Type& alibi_et,
                     const ov::Shape& alibi_shape,
                     const int32_t* max_context_len,
                     const int32_t* score_aggregation_window,
                     const int32_t* rotated_block_indices,
                     std::size_t rotated_block_count,
                     const int32_t* rotation_deltas,
                     const ov::Shape& rotation_deltas_shape,
                     const void* rotation_trig_lut,
                     const ov::element::Type& trig_lut_et,
                     const ov::Shape& trig_lut_shape,
                     const void* xattention_threshold,
                     const ov::element::Type& xattention_threshold_et,
                     const int32_t* xattention_block_size,
                     const int32_t* xattention_stride,
                     const void* sinks,
                     const ov::element::Type& sinks_et,
                     const int32_t* adaptive_rkv_start_size,
                     const int32_t* adaptive_rkv_evictable_sizes,
                     const int32_t* adaptive_rkv_diversity_block_set_indices,
                     const int32_t* adaptive_rkv_diversity_block_set_indices_begins,
                     const ov::Shape& query_shape,
                     const ov::Shape& key_shape,
                     const ov::Shape& value_shape,
                     const ov::Shape& key_cache_shape,
                     const ov::Shape& value_cache_shape,
                     const ov::Shape& past_lens_shape,
                     const ov::Shape& subseq_shape) {
    OPENVINO_ASSERT(cache_manager != nullptr, "PagedAttention reference: cache_manager is null");
    OPENVINO_ASSERT(query != nullptr && key != nullptr && value != nullptr, "PagedAttention reference: null Q/K/V");
    OPENVINO_ASSERT(out != nullptr, "PagedAttention reference: output is null");

    // Currently ignored output (with shape_infer)
    (void)out_aux;

    // Currently ignored inputs
    (void)xattention_threshold;
    (void)xattention_threshold_et;
    (void)xattention_block_size;
    (void)xattention_stride;
    (void)sinks;
    (void)sinks_et;
    (void)adaptive_rkv_start_size;
    (void)adaptive_rkv_evictable_sizes;
    (void)adaptive_rkv_diversity_block_set_indices;
    (void)adaptive_rkv_diversity_block_set_indices_begins;

    OPENVINO_ASSERT(query_shape.size() == 2 && key_shape.size() == 2 && value_shape.size() == 2,
                    "PagedAttention reference: expected Q/K/V rank 2");
    OPENVINO_ASSERT(past_lens_shape.size() == 1 && subseq_shape.size() == 1,
                    "PagedAttention reference: expected past_lens/subsequence_begins rank 1");

    const std::size_t batch_tokens = static_cast<std::size_t>(query_shape[0]);
    const std::size_t query_features = static_cast<std::size_t>(query_shape[1]);
    const std::size_t key_features = static_cast<std::size_t>(key_shape[1]);
    const std::size_t value_features = static_cast<std::size_t>(value_shape[1]);

    const std::size_t seq_count = static_cast<std::size_t>(past_lens_shape[0]);
    const std::size_t subseq_count = static_cast<std::size_t>(subseq_shape[0]);

    // Register operator once per node and copy init cache.
    cache_manager->ensure_operator(node_key,
                                   key_cache_init,
                                   value_cache_init,
                                   key_cache_shape,
                                   value_cache_shape,
                                   block_indices_init,
                                   block_indices_count,
                                   block_indices_begins_init,
                                   block_indices_begins_count,
                                   past_lens,
                                   seq_count);

    const std::size_t head_size = cache_manager->key_head_size(node_key);
    const std::size_t kv_heads = cache_manager->num_kv_heads(node_key);
    const std::size_t v_head_size = cache_manager->value_head_size(node_key);

    OPENVINO_ASSERT(head_size > 0 && kv_heads > 0 && v_head_size > 0, "PagedAttention reference: invalid cache layout");
    OPENVINO_ASSERT(key_features == kv_heads * head_size,
                    "PagedAttention reference: key feature dim mismatch with cache layout");
    OPENVINO_ASSERT(value_features == kv_heads * v_head_size,
                    "PagedAttention reference: value feature dim mismatch with cache layout");
    OPENVINO_ASSERT((query_features % head_size) == 0,
                    "PagedAttention reference: query features not divisible by head_size");

    const std::size_t q_heads = query_features / head_size;
    OPENVINO_ASSERT((q_heads % kv_heads) == 0,
                    "PagedAttention reference: expected num_heads to be divisible by num_kv_heads (GQA)");
    const std::size_t group = q_heads / kv_heads;

    const std::size_t out_features = q_heads * v_head_size;
    const float scale_f = detail::read_scalar_as_f32(scale, scale_et);
    const int32_t sliding_window_i = sliding_window ? sliding_window[0] : 0;
    const int32_t max_context_i = max_context_len ? max_context_len[0] : 0;
    const int32_t score_window_i = score_aggregation_window ? score_aggregation_window[0] : 0;
    const std::size_t alibi_len = alibi_shape.empty() ? 0 : static_cast<std::size_t>(alibi_shape[0]);

    // Initialize per-sequence view of current lengths.
    cache_manager->begin_step(node_key, past_lens, seq_count);

    // Parse token-to-sequence partition.
    const auto seq_begins = detail::parse_subsequence_ranges(subsequence_begins, subseq_count, seq_count, batch_tokens);

    // Prepare mapping for rotated blocks (block_id -> rotated_index).
    std::unordered_map<int32_t, int32_t> rotated_map;
    rotated_map.reserve(rotated_block_count);
    for (std::size_t i = 0; i < rotated_block_count; ++i) {
        rotated_map.emplace(rotated_block_indices[i], static_cast<int32_t>(i));
    }

    const bool has_trig = (rotation_trig_lut != nullptr) && (!trig_lut_shape.empty());
    const std::size_t trig_rows = trig_lut_shape.size() == 2   ? static_cast<std::size_t>(trig_lut_shape[0])
                                  : trig_lut_shape.size() == 1 ? 1
                                                               : 0;
    const bool deltas_is_2d = rotation_deltas_shape.size() == 2;
    const std::size_t deltas_stride = deltas_is_2d ? static_cast<std::size_t>(rotation_deltas_shape[1]) : 0;

    // out_scores: concatenation of [past_len + new_len] for each sequence.
    std::vector<float> scores_acc;
    if (out_scores != nullptr) {
        std::size_t total = 0;
        for (std::size_t s = 0; s < seq_count; ++s) {
            const std::size_t past = past_lens ? static_cast<std::size_t>(past_lens[s]) : 0;
            const std::size_t new_len = seq_begins[s + 1] - seq_begins[s];
            total += past + new_len;
        }
        scores_acc.assign(total, 0.f);
    }

    std::vector<float> logits;
    std::vector<float> out_head;
    std::vector<float> key_buf;  // used for rotary

    // Prefix offsets for out_scores concatenation
    std::vector<std::size_t> score_prefix(seq_count + 1, 0);
    if (!scores_acc.empty()) {
        for (std::size_t s = 0; s < seq_count; ++s) {
            const std::size_t past = past_lens ? static_cast<std::size_t>(past_lens[s]) : 0;
            const std::size_t new_len = seq_begins[s + 1] - seq_begins[s];
            score_prefix[s + 1] = score_prefix[s] + past + new_len;
        }
    }

    for (std::size_t s = 0; s < seq_count; ++s) {
        const std::size_t t_begin = seq_begins[s];
        const std::size_t t_end = seq_begins[s + 1];
        if (t_begin >= t_end) {
            continue;
        }
        OPENVINO_ASSERT(t_end <= batch_tokens, "PagedAttention reference: subsequence range exceeds batch_tokens");

        const std::int32_t past = past_lens ? past_lens[s] : 0;
        OPENVINO_ASSERT(past >= 0, "PagedAttention reference: negative past_lens");
        const std::size_t new_len = t_end - t_begin;

        // Base offset for out_scores for this sequence (concatenation order is sequence order).
        const std::size_t score_base = score_prefix[s];

        for (std::size_t i = 0; i < new_len; ++i) {
            const std::size_t token = t_begin + i;
            const std::int32_t qpos = past + static_cast<std::int32_t>(i);

            // Append this token's KV into the cache.
            const T* krow = key + token * key_features;
            const T* vrow = value + token * value_features;
            cache_manager->write_token_kv<T>(node_key, s, qpos, krow, vrow);

            // Determine attention window.
            std::int32_t start = 0;
            if (max_context_i > 0) {
                start = std::max(start, qpos + 1 - max_context_i);
            }
            if (sliding_window_i > 0) {
                start = std::max(start, qpos + 1 - sliding_window_i);
            }
            if (start < 0) {
                start = 0;
            }
            const std::int32_t end = qpos;
            const std::size_t ctx_len = (end >= start) ? static_cast<std::size_t>(end - start + 1) : 0;
            if (ctx_len == 0) {
                // No context (shouldn't happen for causal), output zeros
                std::fill(out + token * out_features, out + (token + 1) * out_features, T(0));
                continue;
            }

            // For out_scores aggregation, decide whether to include this query
            const bool include_in_scores =
                (scores_acc.empty())
                    ? false
                    : (score_window_i <= 0 ? true : (static_cast<std::int32_t>(new_len - i) <= score_window_i));

            const T* qrow = query + token * query_features;

            for (std::size_t h = 0; h < q_heads; ++h) {
                const std::size_t kvh = h / group;
                const float slope = (alibi_slopes == nullptr || alibi_len == 0)
                                        ? 0.f
                                        : detail::read_at_as_f32(alibi_slopes,
                                                                 alibi_et,
                                                                 (alibi_len == 1) ? 0 : std::min(h, alibi_len - 1));

                const T* qptr = qrow + h * head_size;

                logits.assign(ctx_len, 0.f);
                out_head.assign(v_head_size, 0.f);

                for (std::size_t t = 0; t < ctx_len; ++t) {
                    const std::int32_t kpos = start + static_cast<std::int32_t>(t);
                    ov::reference::paged_attention_cache::PagedCacheManager::TokenAddress addr;
                    const bool ok = cache_manager->resolve_token(node_key, s, kpos, addr);
                    if (!ok) {
                        logits[t] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    const T* kptr = cache_manager->key_ptr<T>(node_key, addr, kvh);
                    if (kptr == nullptr) {
                        logits[t] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    // Optional rotary re-rotation for specific blocks
                    float dot = 0.f;
                    const auto it = rotated_map.find(addr.block);
                    if (has_trig && rotation_deltas != nullptr && it != rotated_map.end()) {
                        const int32_t rot_idx = it->second;
                        std::size_t trig_row = 0;
                        if (deltas_is_2d) {
                            const std::size_t off = static_cast<std::size_t>(addr.offset);
                            const std::size_t di = static_cast<std::size_t>(rot_idx) * deltas_stride + off;
                            const std::size_t deltas_total = static_cast<std::size_t>(rotation_deltas_shape[0]) *
                                                             static_cast<std::size_t>(rotation_deltas_shape[1]);
                            const int32_t raw = (di < deltas_total) ? rotation_deltas[di] : 0;
                            trig_row = raw < 0 ? 0 : static_cast<std::size_t>(raw);
                        } else {
                            const std::size_t deltas_total = rotation_deltas_shape.size() == 1
                                                                 ? static_cast<std::size_t>(rotation_deltas_shape[0])
                                                                 : 0;
                            const std::size_t di = static_cast<std::size_t>(rot_idx);
                            const int32_t raw = (di < deltas_total) ? rotation_deltas[di] : 0;
                            trig_row = raw < 0 ? 0 : static_cast<std::size_t>(raw);
                        }
                        if (trig_rows > 0) {
                            trig_row = std::min(trig_row, trig_rows - 1);
                        }

                        key_buf.assign(head_size, 0.f);
                        for (std::size_t d = 0; d < head_size; ++d) {
                            key_buf[d] = static_cast<float>(kptr[d]);
                        }
                        detail::apply_rotary_inplace(key_buf, rotation_trig_lut, trig_lut_et, trig_row, head_size);
                        for (std::size_t d = 0; d < head_size; ++d) {
                            dot += static_cast<float>(qptr[d]) * key_buf[d];
                        }
                    } else {
                        for (std::size_t d = 0; d < head_size; ++d) {
                            dot += static_cast<float>(qptr[d]) * static_cast<float>(kptr[d]);
                        }
                    }

                    float l = dot * scale_f;
                    if (alibi_slopes != nullptr) {
                        // Typical ALiBi: bias proportional to distance (key_pos - query_pos)
                        l += slope * static_cast<float>(kpos - qpos);
                    }
                    logits[t] = l;
                }

                detail::softmax_inplace(logits);

                for (std::size_t t = 0; t < ctx_len; ++t) {
                    const std::int32_t kpos = start + static_cast<std::int32_t>(t);
                    ov::reference::paged_attention_cache::PagedCacheManager::TokenAddress addr;
                    if (!cache_manager->resolve_token(node_key, s, kpos, addr)) {
                        continue;
                    }
                    const T* vptr = cache_manager->value_ptr<T>(node_key, addr, kvh);
                    if (!vptr) {
                        continue;
                    }
                    const float w = logits[t];
                    for (std::size_t d = 0; d < v_head_size; ++d) {
                        out_head[d] += w * static_cast<float>(vptr[d]);
                    }

                    if (include_in_scores) {
                        // Accumulate score per key position (sum over heads and query tokens in the window)
                        // Index within this sequence's concatenated [past + new] timeline:
                        // past occupies [0, past-1], new tokens occupy [past, past+new_len-1]
                        const std::size_t idx = score_base + static_cast<std::size_t>(kpos);
                        if (idx < scores_acc.size()) {
                            scores_acc[idx] += w;
                        }
                    }
                }

                T* out_ptr = out + token * out_features + h * v_head_size;
                for (std::size_t d = 0; d < v_head_size; ++d) {
                    out_ptr[d] = static_cast<T>(out_head[d]);
                }
            }
        }
    }

    if (out_scores != nullptr) {
        for (std::size_t i = 0; i < scores_acc.size(); ++i) {
            out_scores[i] = static_cast<T>(scores_acc[i]);
        }
    }
}

}  // namespace reference
}  // namespace ov
