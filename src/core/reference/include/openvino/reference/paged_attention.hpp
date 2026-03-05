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
#include "openvino/reference/adaptive_rkv_diversity.hpp"
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

/// Softmax with optional attention-sink support
/// When sink_val is not nullptr, the sink acts as a virtual extra attention score:
/// it participates in the max and exp-sum but produces no output weight
inline void softmax_inplace(std::vector<float>& scores, const float* sink_val = nullptr) {
    if (scores.empty()) {
        return;
    }
    float m = *std::max_element(scores.begin(), scores.end());
    if (sink_val != nullptr) {
        m = std::max(m, *sink_val);
    }
    float sum = 0.f;
    for (float& s : scores) {
        s = std::exp(s - m);
        sum += s;
    }
    if (sink_val != nullptr) {
        sum += std::exp(*sink_val - m);
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
    // Split-half (LLaMA-style) pairing: first half [0..half) paired with second half [half..head_size)
    // Trig LUT row: [cos_0..cos_{half-1}, sin_0..sin_{half-1}]
    for (std::size_t d = 0; d < half; ++d) {
        const float x0 = vec[d];
        const float x1 = vec[d + half];
        const float c = read_at_as_f32(trig_lut, trig_et, row_base + d);
        const float s = read_at_as_f32(trig_lut, trig_et, row_base + half + d);
        vec[d] = x0 * c - x1 * s;
        vec[d + half] = x0 * s + x1 * c;
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

    // Clamp and monotonic fix
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
// Supports: GQA, ALiBi, RoPE re-rotation (split-half / LLaMA-style), attention sinks,
//           xattention sparse prefill, and adaptive RKV diversity scoring

template <typename T>
void paged_attention(std::uintptr_t node_key,
                     ov::reference::paged_attention_cache::PagedCacheManager* cache_manager,
                     T* out,
                     T* out_scores,
                     T* out_diversity_scores,
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
                     std::size_t score_aggregation_window_count,
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

    // These inputs tell CPU/GPU kernels where the physical memory blocks for adaptive RKV are
    // The reference doesn't use them - it reads key data directly from the cache manager
    (void)adaptive_rkv_diversity_block_set_indices;
    (void)adaptive_rkv_diversity_block_set_indices_begins;

    // Sinks: each head has a scalar value treated as a virtual extra token in the softmax
    // (it shifts the max/denominator but does not contribute to the weighted output)
    const bool has_sinks = (sinks != nullptr);

    // Xattention: dynamic sparse prefill via strided Q*K dot products, grouped into blocks, with low-importance blocks
    // masked out
    const bool has_xattn = (xattention_threshold != nullptr);

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

    // Register operator once per node and copy init cache
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
    const std::size_t alibi_len = alibi_shape.empty() ? 0 : static_cast<std::size_t>(alibi_shape[0]);

    // Initialize per-sequence view of current lengths
    cache_manager->begin_step(node_key, past_lens, seq_count);

    // Parse token-to-sequence partition
    const auto seq_begins = detail::parse_subsequence_ranges(subsequence_begins, subseq_count, seq_count, batch_tokens);

    // Prepare mapping for rotated blocks (block_id -> rotated_index)
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

    // out_scores: concatenation of [past_len + new_len] for each sequence
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
    std::vector<float> key_buf;  // re-used for rotary re-rotation

    // Sink: read the per-head scalar values from the [1,H,1,1] input
    std::vector<float> sink_vals;
    if (has_sinks) {
        sink_vals.resize(q_heads, 0.f);
        for (std::size_t h = 0; h < q_heads; ++h) {
            sink_vals[h] = detail::read_at_as_f32(sinks, sinks_et, h);
        }
    }

    // --- Xattention: build block-level sparse mask per head ---
    // xattn_mask[h][q_block][k_block] = true means "keep this block pair"
    // Only activated for multi-token (prefill), single sequence
    const int32_t xattn_block_sz = (xattention_block_size != nullptr) ? xattention_block_size[0] : 0;
    const int32_t xattn_stride = (xattention_stride != nullptr) ? xattention_stride[0] : 0;
    // xattn_mask: [q_heads][num_q_blocks][num_k_blocks]
    std::vector<std::vector<std::vector<bool>>> xattn_mask;

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

        // Per-sequence score aggregation window: shape [] broadcasts to all sequences; shape [B_seq]
        // selects per-sequence value.  Negative → unbounded (all tokens), 0 → disabled.
        const std::size_t saw_idx =
            (score_aggregation_window_count > 1) ? std::min(s, score_aggregation_window_count - 1) : 0;
        const int32_t score_window_i = score_aggregation_window ? score_aggregation_window[saw_idx] : 0;

        // --- Build xattention sparse mask for this sequence (prefill only) ---
        // xattn_mask[h][q_blk][k_blk] = true => keep; false => skip
        // Only activate when: xattn enabled, multi-token (new_len > 1), single sequence
        // When past > 0, the mask covers new tokens only; cached tokens are always attended to
        const bool do_xattn = has_xattn && new_len > 1 && seq_count == 1 && xattn_block_sz > 0 && xattn_stride > 0;
        xattn_mask.clear();
        if (do_xattn) {
            // Phase 1: strided Q*K dot products over the new tokens only
            const float threshold_f = detail::read_scalar_as_f32(xattention_threshold, xattention_threshold_et);
            // mask covers only the new tokens; past cached tokens are always attended to
            const std::size_t total_len = new_len;
            const std::size_t num_q_blocks =
                (total_len + static_cast<std::size_t>(xattn_block_sz) - 1) / static_cast<std::size_t>(xattn_block_sz);
            const std::size_t num_k_blocks = num_q_blocks;
            const std::size_t q_strided =
                (total_len + static_cast<std::size_t>(xattn_stride) - 1) / static_cast<std::size_t>(xattn_stride);
            const std::size_t k_strided = q_strided;
            const std::size_t num_per_block =
                static_cast<std::size_t>(xattn_block_sz) / static_cast<std::size_t>(xattn_stride);

            xattn_mask.resize(q_heads);
            for (std::size_t h = 0; h < q_heads; ++h) {
                const std::size_t kvh = h / group;

                // Compute strided attention matrix [q_strided * k_strided]
                // For each stride offset, average the Q·K scores
                std::vector<float> attn_strided(q_strided * k_strided, 0.f);

                for (int32_t off = 0; off < xattn_stride; ++off) {
                    for (std::size_t qi = 0; qi < q_strided; ++qi) {
                        // Query token index: we sample the (xattn_stride-1-off + qi*stride)-th Q
                        const int64_t q_tok_idx = static_cast<int64_t>(xattn_stride) - 1 - off +
                                                  static_cast<int64_t>(qi) * static_cast<int64_t>(xattn_stride);
                        if (q_tok_idx < 0 || static_cast<std::size_t>(q_tok_idx) >= total_len)
                            continue;
                        const std::size_t q_abs = t_begin + static_cast<std::size_t>(q_tok_idx);
                        const T* qptr = query + q_abs * query_features + h * head_size;

                        // Causal: attend up to qi+1 strided positions
                        const std::size_t k_causal = qi + 1;
                        for (std::size_t ki = 0; ki < std::min(k_causal, k_strided); ++ki) {
                            const int64_t k_tok_idx = static_cast<int64_t>(off) +
                                                      static_cast<int64_t>(ki) * static_cast<int64_t>(xattn_stride);
                            if (k_tok_idx < 0 || static_cast<std::size_t>(k_tok_idx) >= total_len)
                                continue;

                            // Read key from cache or from current input
                            const std::size_t k_pos = static_cast<std::size_t>(k_tok_idx);
                            // all keys for strided estimation come from the new input tokens
                            const T* kptr_raw = key + (t_begin + k_pos) * key_features + kvh * head_size;
                            float dot = 0.f;
                            for (std::size_t d = 0; d < head_size; ++d) {
                                dot += static_cast<float>(qptr[d]) * static_cast<float>(kptr_raw[d]);
                            }
                            attn_strided[qi * k_strided + ki] += dot;
                        }
                    }
                }

                // Scale and causal softmax on strided attention
                const float xscale = scale_f / static_cast<float>(xattn_stride);
                for (std::size_t qi = 0; qi < q_strided; ++qi) {
                    const std::size_t k_causal = qi + 1;
                    float* row = attn_strided.data() + qi * k_strided;
                    // Apply scale
                    for (std::size_t ki = 0; ki < k_strided; ++ki) {
                        row[ki] *= xscale;
                    }
                    // Causal mask: set positions after causal boundary to -inf
                    for (std::size_t ki = k_causal; ki < k_strided; ++ki) {
                        row[ki] = -std::numeric_limits<float>::infinity();
                    }
                    // Softmax over this row
                    float m = -std::numeric_limits<float>::infinity();
                    for (std::size_t ki = 0; ki < k_strided; ++ki)
                        m = std::max(m, row[ki]);
                    float sm = 0.f;
                    for (std::size_t ki = 0; ki < k_strided; ++ki) {
                        row[ki] = std::exp(row[ki] - m);
                        sm += row[ki];
                    }
                    if (sm > 0.f) {
                        float inv = 1.f / sm;
                        for (std::size_t ki = 0; ki < k_strided; ++ki)
                            row[ki] *= inv;
                    }
                }

                // Phase 2: block aggregation - sum num_per_block * num_per_block windows
                std::vector<float> block_sums(num_q_blocks * num_k_blocks, 0.f);
                for (std::size_t qb = 0; qb < num_q_blocks; ++qb) {
                    for (std::size_t kb = 0; kb < num_k_blocks; ++kb) {
                        float bsum = 0.f;
                        for (std::size_t dq = 0; dq < num_per_block; ++dq) {
                            const std::size_t qi = qb * num_per_block + dq;
                            if (qi >= q_strided)
                                continue;
                            for (std::size_t dk = 0; dk < num_per_block; ++dk) {
                                const std::size_t ki = kb * num_per_block + dk;
                                if (ki >= k_strided)
                                    continue;
                                bsum += attn_strided[qi * k_strided + ki];
                            }
                        }
                        block_sums[qb * num_k_blocks + kb] = bsum;
                    }
                }

                // Phase 3: threshold masking - greedy top-block selection
                xattn_mask[h].resize(num_q_blocks, std::vector<bool>(num_k_blocks, false));
                for (std::size_t qb = 0; qb < num_q_blocks; ++qb) {
                    const float* brow = block_sums.data() + qb * num_k_blocks;
                    float total_sum = 0.f;
                    for (std::size_t kb = 0; kb <= qb && kb < num_k_blocks; ++kb) {
                        total_sum += brow[kb];
                    }
                    const float required = total_sum * threshold_f;

                    // Build (value, index) pairs
                    std::vector<std::pair<float, std::size_t>> vals;
                    vals.reserve(num_k_blocks);
                    for (std::size_t kb = 0; kb < num_k_blocks; ++kb) {
                        vals.emplace_back(brow[kb], kb);
                    }
                    // Block 0 (first column) and diagonal (qb) always selected
                    // Swap them to positions 0 and 1, then sort rest descending
                    if (qb > 1 && qb < vals.size()) {
                        std::swap(vals[1], vals[qb]);
                    }
                    if (vals.size() > 2) {
                        std::sort(vals.begin() + 2, vals.end(), [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                    }

                    // Cumulative sum selection
                    float cumsum = 0.f;
                    for (std::size_t i = 0; i < vals.size(); ++i) {
                        bool keep = false;
                        if (i == 0) {
                            // First column always kept
                            keep = true;
                        } else if (i == 1 && qb >= 1) {
                            // Diagonal always kept
                            keep = true;
                            cumsum = vals[0].first;
                        } else {
                            keep = (cumsum < required);
                            cumsum += vals[i - 1].first;
                        }
                        // Causal: only keep blocks where k_block <= q_block
                        if (vals[i].second > qb)
                            keep = false;
                        xattn_mask[h][qb][vals[i].second] = keep;
                    }
                }
            }
        }

        // Base offset for out_scores for this sequence (concatenation order is sequence order)
        const std::size_t score_base = score_prefix[s];

        for (std::size_t i = 0; i < new_len; ++i) {
            const std::size_t token = t_begin + i;
            const std::int32_t qpos = past + static_cast<std::int32_t>(i);

            // Append this token's KV into the cache
            const T* krow = key + token * key_features;
            const T* vrow = value + token * value_features;
            cache_manager->write_token_kv<T>(node_key, s, qpos, krow, vrow);

            // Determine attention window
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

            // For out_scores aggregation, decide whether to include this query.
            // score_window_i == 0 → disabled: no accumulation, output 1 stays zero.
            // score_window_i  > 0 → only the last score_window_i tokens contribute.
            // score_window_i  < 0 → all tokens contribute (unbounded window).
            const bool include_in_scores =
                (scores_acc.empty() || score_window_i == 0)
                    ? false
                    : (score_window_i < 0 ? true : (static_cast<std::int32_t>(new_len - i) <= score_window_i));

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

                    // Re-apply RoPE to pre-existing cached positions only; new tokens are
                    // written with unrotated keys, consistent with the CPU kernel
                    float dot = 0.f;
                    const auto it = rotated_map.find(addr.block);
                    if (has_trig && rotation_deltas != nullptr && it != rotated_map.end() && kpos < past) {
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
                        l += slope * static_cast<float>(kpos - qpos);
                    }
                    logits[t] = l;
                }

                // Apply xattention sparse mask: set skipped block pairs to -inf
                // Mask covers new-to-new attention only; kpos < past means a cached key, always included
                if (do_xattn && h < xattn_mask.size()) {
                    for (std::size_t t = 0; t < ctx_len; ++t) {
                        const std::int32_t kpos = start + static_cast<std::int32_t>(t);
                        if (kpos < past)
                            continue;  // past cached tokens always included
                        const std::size_t q_blk = i / static_cast<std::size_t>(xattn_block_sz);
                        const std::size_t k_blk =
                            static_cast<std::size_t>(kpos - past) / static_cast<std::size_t>(xattn_block_sz);
                        if (q_blk < xattn_mask[h].size() && k_blk < xattn_mask[h][q_blk].size()) {
                            if (!xattn_mask[h][q_blk][k_blk]) {
                                logits[t] = -std::numeric_limits<float>::infinity();
                            }
                        }
                    }
                }

                // Softmax with optional attention sink
                const float* sink_ptr = has_sinks ? &sink_vals[h] : nullptr;
                detail::softmax_inplace(logits, sink_ptr);

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
                        // Accumulate attention weight at each key position in the [past..past+new) timeline
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

    // Feed accumulated scores back into the cache manager so that score-based
    // eviction can use them on the next allocation that runs out of free blocks
    if (!scores_acc.empty()) {
        cache_manager->update_attention_scores(node_key, scores_acc.data(), scores_acc.size(), past_lens, seq_count);
    }

    // --- Adaptive RKV diversity computation ---
    // After all attention is done, compute per-block diversity scores for the
    // eviction zone of each sequence using AdaptiveRKVDiversityCalculator
    // The diversity output is a flat buffer: for each sequence, the calculator
    // returns [eviction_size / block_size, eviction_size] and we flatten it
    const bool has_adaptive_rkv = (adaptive_rkv_evictable_sizes != nullptr);
    if (has_adaptive_rkv && out_diversity_scores != nullptr) {
        const std::size_t cache_block_size = cache_manager->block_size(node_key);
        const int32_t start_size_i = adaptive_rkv_start_size ? adaptive_rkv_start_size[0] : 0;
        const std::size_t start_size = (start_size_i < 0) ? 0 : static_cast<std::size_t>(start_size_i);

        std::size_t out_offset = 0;
        for (std::size_t s = 0; s < seq_count; ++s) {
            const int32_t evict_size_i = adaptive_rkv_evictable_sizes[s];
            if (evict_size_i <= 0) {
                continue;
            }
            const std::size_t evict_size = static_cast<std::size_t>(evict_size_i);

            // Total tokens for this sequence (past + newly appended)
            const std::size_t past = past_lens ? static_cast<std::size_t>(past_lens[s]) : 0;
            const std::size_t new_len = seq_begins[s + 1] - seq_begins[s];
            const std::size_t total_tokens = past + new_len;

            if (total_tokens < start_size + evict_size) {
                // Not enough tokens for the eviction zone - skip
                continue;
            }

            // Assemble key data from cache: [kv_heads, total_tokens, head_size]
            std::vector<T> key_data(kv_heads * total_tokens * head_size, T(0));
            for (std::size_t kvh = 0; kvh < kv_heads; ++kvh) {
                for (std::size_t pos = 0; pos < total_tokens; ++pos) {
                    ov::reference::paged_attention_cache::PagedCacheManager::TokenAddress addr;
                    if (cache_manager->resolve_token(node_key, s, static_cast<int32_t>(pos), addr)) {
                        const T* kptr = cache_manager->key_ptr<T>(node_key, addr, kvh);
                        if (kptr) {
                            T* dst = key_data.data() + kvh * total_tokens * head_size + pos * head_size;
                            std::copy(kptr, kptr + head_size, dst);
                        }
                    }
                }
            }

            ov::Shape key_shape_3d = {kv_heads, total_tokens, head_size};
            ov::reference::AdaptiveRKVDiversityCalculator<T> calc(start_size, evict_size, cache_block_size);
            auto diversity = calc.calculate_block_diversity(key_data.data(), key_shape_3d);

            // Flatten [evict_size/block_size][evict_size] into the output buffer
            for (std::size_t b = 0; b < diversity.size(); ++b) {
                for (std::size_t t = 0; t < diversity[b].size(); ++t) {
                    out_diversity_scores[out_offset++] = diversity[b][t];
                }
            }

            // Feed per-block diversity back into the cache manager so adaptive RKV
            // eviction can pick the least diverse block when the pool runs out\n            // We reduce each
            // [evict_size/block_size][evict_size] row to a single\n            // per-block mean diversity value
            const std::size_t n_evict_blocks = diversity.size();
            std::vector<float> per_block_div(n_evict_blocks, 0.f);
            for (std::size_t b = 0; b < n_evict_blocks; ++b) {
                float sum = 0.f;
                for (std::size_t t = 0; t < diversity[b].size(); ++t) {
                    sum += static_cast<float>(diversity[b][t]);
                }
                per_block_div[b] = diversity[b].empty() ? 0.f : sum / static_cast<float>(diversity[b].size());
            }
            // start_block_offset is the block index after the start area
            const std::size_t start_blk_off = start_size / cache_block_size;
            cache_manager->update_diversity_scores(node_key, s, per_block_div.data(), n_evict_blocks, start_blk_off);
        }
    }
}

}  // namespace reference
}  // namespace ov
