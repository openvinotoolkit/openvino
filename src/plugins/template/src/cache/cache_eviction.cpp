// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/template/cache/cache_eviction.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "openvino/core/type/float16.hpp"

namespace ov::template_plugin::cache {

// -------- helpers --------

namespace {
template <typename T>
inline const T* as_ptr(const ov::Tensor& t) {
    return reinterpret_cast<const T*>(t.data());
}

inline void assert_3d_scores(const ov::Tensor& t, const char* who) {
    const auto& s = t.get_shape();
    OPENVINO_ASSERT(s.size() == 3, who, ": expected [T_step, H, L_valid], got rank ", s.size());
    OPENVINO_ASSERT(s[1] > 0 && s[2] > 0, who, ": invalid H or L_valid");
}

inline size_t logical_blocks_from_token_length(size_t tokens, size_t block_size) {
    return (tokens + block_size - 1) / block_size;
}

// Accumulate per-token vector from a [T_step, H, L_valid] score tensor:
//   - sum across heads
//   - max-pool across last 'pool_T' query tokens (SnapKV-like)
std::vector<double> reduce_to_per_token_scores(const ov::Tensor& t, size_t pool_T) {
    assert_3d_scores(t, "reduce_to_per_token_scores");
    const auto& s = t.get_shape(); // [T, H, L]
    const size_t T = s[0], H = s[1], L = s[2];
    pool_T = std::min(pool_T, T);
    const size_t t_begin = T - pool_T;

    std::vector<double> out(L, 0.0);

    auto et = t.get_element_type();
    if (et == ov::element::f32) {
        const float* base = as_ptr<float>(t);
        for (size_t j = 0; j < L; ++j) {
            double best = 0.0;
            for (size_t tt = t_begin; tt < T; ++tt) {
                const float* row = base + (tt * H * L) + j; // we'll stride by L across heads
                double sum_h = 0.0;
                for (size_t h = 0; h < H; ++h) sum_h += row[h * L];
                best = std::max(best, sum_h);
            }
            out[j] = best;
        }
    } else if (et == ov::element::f16) {
        const ov::float16* base = as_ptr<ov::float16>(t);
        for (size_t j = 0; j < L; ++j) {
            double best = 0.0;
            for (size_t tt = t_begin; tt < T; ++tt) {
                const ov::float16* row = base + (tt * H * L) + j;
                double sum_h = 0.0;
                for (size_t h = 0; h < H; ++h) sum_h += static_cast<float>(row[h * L]);
                best = std::max(best, sum_h);
            }
            out[j] = best;
        }
    } else if (et == ov::element::bf16) {
        const ov::bfloat16* base = as_ptr<ov::bfloat16>(t);
        for (size_t j = 0; j < L; ++j) {
            double best = 0.0;
            for (size_t tt = t_begin; tt < T; ++tt) {
                const ov::bfloat16* row = base + (tt * H * L) + j;
                double sum_h = 0.0;
                for (size_t h = 0; h < H; ++h) sum_h += static_cast<float>(row[h * L]);
                best = std::max(best, sum_h);
            }
            out[j] = best;
        }
    } else {
        OPENVINO_ASSERT(false, "Unsupported dtype in reduce_to_per_token_scores: ", et);
    }

    return out;
}

} // namespace

// ============ EvictionScoreManager ============

void EvictionScoreManager::register_new_token_scores(
    const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers,
    const std::set<size_t>& skipped_logical_block_ids) {
    OPENVINO_ASSERT(attention_scores_for_all_decoder_layers.size() == m_num_decoder_layers,
                    "EvictionScoreManager: mismatched num layers");

    for (size_t layer = 0; layer < m_num_decoder_layers; ++layer) {
        const auto& tens = attention_scores_for_all_decoder_layers[layer];
        assert_3d_scores(tens, "EvictionScoreManager::register_new_token_scores");
        const auto& s = tens.get_shape(); // [T_step, H, L_valid]
        const size_t L_valid = s[2];

        // Reduce to per-token vector (length L_valid)
        std::vector<double> step_scores = reduce_to_per_token_scores(tens, m_max_pool_window_size);

        // Ensure internal buffers are at least L_valid long (extend with zeros if the sequence grew)
        auto& dst = m_scores[layer];
        auto& cnt = m_cache_counter[layer];
        if (dst.size() < L_valid) {
            dst.resize(L_valid, 0.0);
            cnt.resize(L_valid, 0);
        }

        // Apply "ignore first N blocks"
        const size_t ignore_tokens = m_ignore_first_n_blocks * m_block_size;
        if (step_scores.size() > ignore_tokens) {
            // Zero out ignored head
            std::fill(step_scores.begin(), step_scores.begin() + std::min(ignore_tokens, step_scores.size()), 0.0);
        }

        // Aggregate with potential skips (for sparse prefill)
        if (!skipped_logical_block_ids.empty()) {
            add_with_skips(dst, step_scores, skipped_logical_block_ids);
            // Increase counters for tokens that were not skipped
            // (We can't easily do per-token here without reconstructing indices; do a second pass mirroring add_with_skips)
            const size_t dst_tokens = dst.size();
            const size_t total_blocks = logical_blocks_from_token_length(dst_tokens, m_block_size);
            const size_t skipped = skipped_logical_block_ids.size();
            OPENVINO_ASSERT(step_scores.size() + skipped * m_block_size == dst_tokens,
                            "EvictionScoreManager: src/dst length mismatch with skips");
            size_t src_pos = 0;
            for (size_t b = 0; b < total_blocks; ++b) {
                const bool skip = skipped_logical_block_ids.count(b) != 0;
                const size_t begin = b * m_block_size;
                const size_t end = std::min(begin + m_block_size, dst_tokens);
                if (skip) {
                    // counters unchanged
                } else {
                    for (size_t i = begin; i < end; ++i, ++src_pos) {
                        if (step_scores[src_pos] != 0.0) cnt[i] += 1;
                    }
                }
            }
        } else {
            // Simple element-wise add and increase counters
            const size_t N = std::min(dst.size(), step_scores.size());
            for (size_t i = 0; i < N; ++i) {
                dst[i] += step_scores[i];
                if (step_scores[i] != 0.0) cnt[i] += 1;
            }
        }

        // If SUM vs NORM_SUM is applied later at readout, we don't normalize here.
        // (Normalization happens when building eviction scores.)
    }
}

void EvictionScoreManager::remove_scores(const std::vector<std::size_t>& evicted_block_indices,
                                         size_t decoder_layer_idx) {
    OPENVINO_ASSERT(decoder_layer_idx < m_scores.size(), "EvictionScoreManager::remove_scores: bad layer");
    if (evicted_block_indices.empty()) return;

    auto& vec = m_scores[decoder_layer_idx];
    auto& cnt = m_cache_counter[decoder_layer_idx];
    const size_t N = vec.size();

    // Erase token ranges corresponding to blocks; erase from high to low to keep indices valid
    std::vector<std::size_t> sorted = evicted_block_indices;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        const size_t b = *it;
        const size_t begin = b * m_block_size;
        if (begin >= N) continue;
        const size_t end = std::min(begin + m_block_size, N);
        vec.erase(vec.begin() + begin, vec.begin() + end);
        cnt.erase(cnt.begin() + begin, cnt.begin() + end);
    }
}

void EvictionScoreManager::add_with_skips(std::vector<double>& dst,
                                          const std::vector<double>& src,
                                          const std::set<size_t>& skipped_logical_block_ids) const {
    const size_t dst_tokens = dst.size();
    const size_t total_blocks = logical_blocks_from_token_length(dst_tokens, m_block_size);
    const size_t expected_src_tokens = dst_tokens - skipped_logical_block_ids.size() * m_block_size;
    OPENVINO_ASSERT(src.size() == expected_src_tokens,
                    "add_with_skips: src length mismatch; expected ", expected_src_tokens, ", got ", src.size());

    size_t src_pos = 0;
    for (size_t b = 0; b < total_blocks; ++b) {
        const bool skip = skipped_logical_block_ids.count(b) != 0;
        const size_t begin = b * m_block_size;
        const size_t end = std::min(begin + m_block_size, dst_tokens);
        if (skip) {
            continue; // leave dst as-is for this block
        } else {
            for (size_t i = begin; i < end; ++i, ++src_pos) {
                dst[i] += src[src_pos];
            }
        }
    }
}

size_t EvictionScoreManager::get_current_scores_length_in_tokens(size_t layer_idx) const {
    OPENVINO_ASSERT(layer_idx < m_scores.size(), "get_current_scores_length_in_tokens: bad layer");
    return m_scores[layer_idx].size();
}

const std::vector<std::vector<double>>& EvictionScoreManager::get_scores() const {
    return m_scores;
}

const std::vector<std::vector<size_t>>& EvictionScoreManager::get_counters() const {
    return m_cache_counter;
}

// ============ CacheEvictionAlgorithm ============

CacheEvictionAlgorithm::CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config,
                                               size_t block_size,
                                               size_t num_decoder_layers,
                                               size_t max_pool_window_size)
    : m_eviction_config(eviction_config),
      m_block_size(block_size),
      m_num_decoder_layers(num_decoder_layers),
      m_score_manager(block_size,
                      num_decoder_layers,
                      max_pool_window_size,
                      eviction_config.aggregation_mode,
                      eviction_config.get_start_size() / block_size) {}

std::size_t CacheEvictionAlgorithm::get_max_cache_size_after_eviction() const {
    // (Max tokens + at most (block_size-1) partial) as in your description
    return m_eviction_config.get_max_cache_size() + m_block_size - 1;
}

CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range() const {
    // Choose layer 0 as canonical; ranges are identical in token space across layers if lengths are synced.
    return get_evictable_block_range(0);
}

void CacheEvictionAlgorithm::register_new_token_scores(
    const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers,
    const std::set<size_t>& skipped_logical_block_ids) {
    m_score_manager.register_new_token_scores(attention_scores_for_all_decoder_layers, skipped_logical_block_ids);
}

void CacheEvictionAlgorithm::register_new_token_scores(
    const AttentionScoresForEachDecoderLayer& attention_scores_across_decoder_layers_for_current_sequence) {
    std::set<size_t> empty_skipped;
    register_new_token_scores(attention_scores_across_decoder_layers_for_current_sequence, empty_skipped);
}

std::vector<std::set<std::size_t>> CacheEvictionAlgorithm::evict_logical_blocks() {
    std::vector<std::set<std::size_t>> result(m_num_decoder_layers);

    for (size_t layer = 0; layer < m_num_decoder_layers; ++layer) {
        const auto range = get_evictable_block_range(layer);
        if (range == CacheEvictionRange::invalid()) continue;

        const size_t need = get_num_blocks_to_evict(layer);
        const size_t can  = get_num_evictable_blocks(layer);
        const size_t num  = std::min(need, can);
        if (num == 0) continue;

        // Scores per evictable block (ordered from 'range.first' block onwards)
        const auto per_block_scores = get_scores_for_all_evictable_blocks(layer);
        auto indices = get_indices_of_blocks_to_evict(per_block_scores, num, /*evictable_begin_block=*/range.first);

        // Erase from tracker; return set
        remove_scores_of_evicted_blocks(indices, layer);
        result[layer].insert(indices.begin(), indices.end());
    }

    return result;
}

std::size_t CacheEvictionAlgorithm::get_num_blocks(std::size_t num_tokens) const {
    return ceil_div(num_tokens, m_block_size);
}

std::size_t CacheEvictionAlgorithm::get_num_blocks_to_evict(size_t decoder_layer_idx) const {
    const size_t tokens_now = m_score_manager.get_current_scores_length_in_tokens(decoder_layer_idx);
    const size_t blocks_now = get_num_blocks(tokens_now);
    const size_t blocks_cap = ceil_div(m_eviction_config.get_max_cache_size(), m_block_size);
    if (blocks_now <= blocks_cap) return 0;
    return blocks_now - blocks_cap;
}

std::size_t CacheEvictionAlgorithm::get_num_evictable_blocks(size_t decoder_layer_idx) const {
    const auto range = get_evictable_block_range(decoder_layer_idx);
    if (range == CacheEvictionRange::invalid()) return 0;
    return (range.second > range.first) ? (range.second - range.first) : 0;
}

CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range(size_t layer_idx) const {
    const size_t tokens_now = m_score_manager.get_current_scores_length_in_tokens(layer_idx);
    if (tokens_now == 0) return CacheEvictionRange::invalid();

    const size_t blocks_now   = get_num_blocks(tokens_now);
    const size_t start_blocks = m_eviction_config.get_start_size()  / m_block_size;
    const size_t recent_blocks= m_eviction_config.get_recent_size() / m_block_size;

    if (blocks_now <= (start_blocks + recent_blocks)) {
        return CacheEvictionRange::invalid();
    }
    const size_t begin = start_blocks;
    const size_t end   = blocks_now > recent_blocks ? (blocks_now - recent_blocks) : begin;
    if (end <= begin) return CacheEvictionRange::invalid();
    return CacheEvictionRange(begin, end); // [begin, end)
}

std::vector<double> CacheEvictionAlgorithm::get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const {
    const auto range = get_evictable_block_range(decoder_layer_idx);
    OPENVINO_ASSERT(!(range == CacheEvictionRange::invalid()), "get_scores_for_all_evictable_blocks: invalid range");

    const auto& all_scores = m_score_manager.get_scores()[decoder_layer_idx];
    const auto& counters   = m_score_manager.get_counters()[decoder_layer_idx];

    const size_t tokens_now = all_scores.size();
    const size_t begin_tok  = range.first  * m_block_size;
    const size_t end_tok    = std::min(range.second * m_block_size, tokens_now);
    const size_t evictable_blocks = range.second - range.first;

    std::vector<double> per_block(evictable_blocks, 0.0);

    // Aggregate per token -> per block (sum). Normalize if NORM_SUM.
    for (size_t tok = begin_tok; tok < end_tok; ++tok) {
        const size_t b = (tok / m_block_size) - range.first;
        double v = all_scores[tok];
        if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
            const size_t c = counters[tok];
            if (c > 0) v /= static_cast<double>(c);
        }
        per_block[b] += v;
    }
    return per_block;
}

std::vector<std::size_t> CacheEvictionAlgorithm::get_indices_of_blocks_to_evict(
    const std::vector<double>& scores_for_each_evictable_block,
    size_t num_blocks_to_evict,
    std::size_t evictable_begin_block) const {

    // Pick the 'num' smallest scores; break ties by lower logical index (stable)
    std::vector<std::pair<double, std::size_t>> pairs;
    pairs.reserve(scores_for_each_evictable_block.size());
    for (size_t i = 0; i < scores_for_each_evictable_block.size(); ++i) {
        pairs.emplace_back(scores_for_each_evictable_block[i], i);
    }
    std::nth_element(pairs.begin(),
                     pairs.begin() + std::min(num_blocks_to_evict, pairs.size()),
                     pairs.end(),
                     [](const auto& a, const auto& b){ return a.first < b.first; });

    const size_t take = std::min(num_blocks_to_evict, pairs.size());
    std::vector<std::size_t> logical_indices;
    logical_indices.reserve(take);
    // We need final deterministic order: ascending logical block id
    std::vector<std::size_t> tmp;
    tmp.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        const size_t local_block = pairs[i].second;
        tmp.push_back(evictable_begin_block + local_block);
    }
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    return tmp;
}

void CacheEvictionAlgorithm::remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices,
                                                             size_t decoder_layer_idx) {
    m_score_manager.remove_scores(evicted_block_indices, decoder_layer_idx);
}

// ============ CacheRotationCalculator ============

CacheRotationCalculator::CacheRotationCalculator(size_t block_size,
                                                 size_t max_context_length,
                                                 size_t kv_head_size,
                                                 double rope_theta)
    : m_block_size(block_size),
      m_head_size(kv_head_size),
      m_rope_sin_lut(max_context_length, std::vector<float>(kv_head_size / 2)),
      m_rope_cos_lut(max_context_length, std::vector<float>(kv_head_size / 2)) {
    // Precompute LUT (basic Llama RoPE)
    for (size_t pos = 0; pos < max_context_length; ++pos) {
        for (size_t i = 0; i < kv_head_size / 2; ++i) {
            const float inv_freq = static_cast<float>(1.0 / std::pow(rope_theta, static_cast<float>(i) / (kv_head_size / 2)));
            const float angle = static_cast<float>(pos) * inv_freq;
            m_rope_sin_lut[pos][i] = std::sin(angle);
            m_rope_cos_lut[pos][i] = std::cos(angle);
        }
    }
}

std::vector<CacheRotationCalculator::BlockRotationData> CacheRotationCalculator::get_rotation_data(
    const std::set<size_t>& /*evicted_block_logical_indices*/,
    size_t /*num_logical_blocks_before_eviction*/,
    bool /*deltas_only*/) {
    // TODO: Implement RoPE-consistent rotation deltas (optional feature)
    return {};
}

const std::vector<std::vector<float>>& CacheRotationCalculator::get_sin_lut() const {
    return m_rope_sin_lut;
}

const std::vector<std::vector<float>>& CacheRotationCalculator::get_cos_lut() const {
    return m_rope_cos_lut;
}

}  // namespace ov::template_plugin::cache
