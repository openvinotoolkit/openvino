// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cache_eviction.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ov {
namespace cache {

// -------- helpers --------

// Treat the LAST dimension of scores tensor as the token axis.
// Aggregate (sum) over all preceding dims to get per-token values.
static std::vector<double> reduce_to_token_axis(const ov::Tensor& t) {
    const auto& shape = t.get_shape();
    if (shape.empty())
        return {};

    size_t token_dim = shape.back();
    std::vector<double> out(token_dim, 0.0);
    if (t.get_size() == 0)
        return out;

    // product of preceding dims
    size_t lead = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
        lead *= shape[i];

    // assume plain row-major
    // layout: [lead, token_dim]
    switch (t.get_element_type()) {
    case ov::element::f16:
    case ov::element::bf16:
    case ov::element::f32:
    case ov::element::f64: {
        // read as float-like
        // convert to f32 in a simple way
        // we'll handle f16/bf16 by casting via float
        // (for brevity; optimize later)
        if (t.get_element_type() == ov::element::f32) {
            const float* p = t.data<const float>();
            for (size_t i = 0; i < lead; ++i) {
                const float* row = p + i * token_dim;
                for (size_t j = 0; j < token_dim; ++j)
                    out[j] += static_cast<double>(row[j]);
            }
        } else if (t.get_element_type() == ov::element::f64) {
            const double* p = t.data<const double>();
            for (size_t i = 0; i < lead; ++i) {
                const double* row = p + i * token_dim;
                for (size_t j = 0; j < token_dim; ++j)
                    out[j] += row[j];
            }
        } else {
            // f16/bf16 path
            const uint16_t* p = t.data<const uint16_t>();
            for (size_t i = 0; i < lead; ++i) {
                const uint16_t* row = p + i * token_dim;
                for (size_t j = 0; j < token_dim; ++j)
                    out[j] += static_cast<double>(ov::float16(row[j]));
            }
        }
        break;
    }
    default:
        // if scores are in other type, reinterpret as float
        const float* p = t.data<const float>();
        for (size_t i = 0; i < lead; ++i) {
            const float* row = p + i * token_dim;
            for (size_t j = 0; j < token_dim; ++j)
                out[j] += static_cast<double>(row[j]);
        }
        break;
    }
    return out;
}

// Max pool over sliding window at the end of prompt (SnapKV-like initializer).
static void max_pool_last_window(std::vector<double>& per_token, size_t window_size) {
    if (window_size == 0 || per_token.empty())
        return;
    const size_t n = per_token.size();
    const size_t start = n > window_size ? (n - window_size) : 0;
    double maxv = 0.0;
    for (size_t i = start; i < n; ++i)
        maxv = std::max(maxv, per_token[i]);
    for (size_t i = start; i < n; ++i)
        per_token[i] = std::max(per_token[i], maxv);
}

// ============ EvictionScoreManager ============

void EvictionScoreManager::register_new_token_scores(const AttentionScoresForEachDecoderLayer& per_layer_scores,
                                                     const std::set<size_t>& skipped_logical_block_ids) {
    OPENVINO_ASSERT(per_layer_scores.size() == m_num_decoder_layers, "EvictionScoreManager: mismatch in layer count");

    for (size_t l = 0; l < m_num_decoder_layers; ++l) {
        // 1) Reduce incoming scores tensor to per-token vector
        std::vector<double> token_scores = reduce_to_token_axis(per_layer_scores[l]);

        // 2) Optional: max-pool last window to stabilize early steps
        if (m_max_pool_window_size > 1) {
            max_pool_last_window(token_scores, m_max_pool_window_size);
        }

        // 3) Grow internal buffers to match current length
        auto& scores = m_scores[l];
        auto& counters = m_cache_counter[l];

        if (scores.size() < token_scores.size()) {
            scores.resize(token_scores.size(), 0.0);
            counters.resize(token_scores.size(), 0);
        }

        // 4) Aggregate with optional "skips"
        add_with_skips(scores, token_scores, skipped_logical_block_ids);

        // 5) Update lifetimes for NORM_SUM mode
        for (size_t i = 0; i < token_scores.size(); ++i) {
            counters[i] += 1;
        }

        // Normalize if requested (lazy: store normalized values back)
        if (m_aggregation_mode == AggregationMode::NORM_SUM) {
            for (size_t i = 0; i < scores.size(); ++i) {
                if (counters[i] > 0)
                    scores[i] /= static_cast<double>(counters[i]);
            }
        }
    }
}

void EvictionScoreManager::remove_scores(const std::vector<std::size_t>& evicted_block_indices, size_t layer_idx) {
    if (layer_idx >= m_num_decoder_layers || m_block_size == 0)
        return;
    auto& scores = m_scores[layer_idx];
    auto& counters = m_cache_counter[layer_idx];
    if (scores.empty())
        return;

    // Remove by erasing whole block-sized chunks (from highest index to lowest)
    std::vector<size_t> sorted = evicted_block_indices;
    std::sort(sorted.begin(), sorted.end());
    for (size_t k = 0; k < sorted.size(); ++k) {
        size_t lb = sorted[sorted.size() - 1 - k];
        size_t off = lb * m_block_size;
        if (off >= scores.size())
            continue;
        size_t len = std::min(m_block_size, scores.size() - off);
        scores.erase(scores.begin() + off, scores.begin() + off + len);
        counters.erase(counters.begin() + off, counters.begin() + off + len);
    }
}

void EvictionScoreManager::add_with_skips(std::vector<double>& dst,
                                          const std::vector<double>& src,
                                          const std::set<size_t>& skipped_logical_block_ids) const {
    if (dst.empty() || src.empty() || m_block_size == 0)
        return;

    // Build a mask of token positions to skip based on logical block ids
    std::vector<char> skip(dst.size(), 0);
    for (size_t lb : skipped_logical_block_ids) {
        size_t off = lb * m_block_size;
        for (size_t j = 0; j < m_block_size && (off + j) < skip.size(); ++j)
            skip[off + j] = 1;
    }

    // Add src into dst for positions not skipped (align from the end if lengths differ)
    const size_t N = std::min(dst.size(), src.size());
    const size_t off_dst = dst.size() - N;
    const size_t off_src = src.size() - N;

    for (size_t i = 0; i < N; ++i) {
        const size_t p = off_dst + i;
        if (!skip[p])
            dst[p] += src[off_src + i];
    }
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
    return m_eviction_config.get_max_cache_size() + m_block_size - 1;
}

CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range() const {
    return get_evictable_block_range(0);
}

void CacheEvictionAlgorithm::register_new_token_scores(const AttentionScoresForEachDecoderLayer& per_layer_scores,
                                                       const std::set<size_t>& skipped_logical_block_ids) {
    m_score_manager.register_new_token_scores(per_layer_scores, skipped_logical_block_ids);
}

void CacheEvictionAlgorithm::register_new_token_scores(
    const AttentionScoresForEachDecoderLayer& per_layer_scores_only) {
    std::set<size_t> none;
    register_new_token_scores(per_layer_scores_only, none);
}

std::vector<std::set<std::size_t>> CacheEvictionAlgorithm::evict_logical_blocks() {
    // Build unified eviction set using averaged block scores across layers over evictable range
    if (m_num_decoder_layers == 0 || m_block_size == 0)
        return {};

    const auto L_tokens = m_score_manager.get_current_scores_length_in_tokens(0);
    if (L_tokens == 0)
        return std::vector<std::set<std::size_t>>(m_num_decoder_layers);  // nothing to evict

    // compute how many blocks to evict (based on layer 0 token length)
    const size_t n_evict_blocks = get_num_blocks_to_evict(0);
    if (n_evict_blocks == 0)
        return std::vector<std::set<std::size_t>>(m_num_decoder_layers);

    // average block scores across layers in the evictable range
    const auto range = get_evictable_block_range(0);
    const size_t lbeg = range.first;
    const size_t lend = range.second;
    if (lend <= lbeg)
        return std::vector<std::set<std::size_t>>(m_num_decoder_layers);

    const size_t num_evictable = lend - lbeg;

    std::vector<double> avg_scores(num_evictable, 0.0);
    for (size_t l = 0; l < m_num_decoder_layers; ++l) {
        auto layer_scores = get_scores_for_all_evictable_blocks(l);
        for (size_t i = 0; i < num_evictable && i < layer_scores.size(); ++i) {
            avg_scores[i] += layer_scores[i];
        }
    }
    for (auto& v : avg_scores)
        v /= static_cast<double>(m_num_decoder_layers);

    // choose lowest n blocks
    auto idx_local = get_indices_of_blocks_to_evict(avg_scores, n_evict_blocks);

    // translate to global logical block ids
    std::vector<size_t> evicted_global;
    evicted_global.reserve(idx_local.size());
    for (auto i : idx_local)
        evicted_global.push_back(lbeg + i);

    // remove from tracking for each layer and return the same set per layer
    std::vector<std::set<std::size_t>> per_layer(m_num_decoder_layers);
    for (size_t l = 0; l < m_num_decoder_layers; ++l) {
        remove_scores_of_evicted_blocks(evicted_global, l);
        per_layer[l].insert(evicted_global.begin(), evicted_global.end());
    }
    return per_layer;
}

std::size_t CacheEvictionAlgorithm::get_num_blocks(std::size_t num_tokens) const {
    return (num_tokens + m_block_size - 1) / m_block_size;
}

std::size_t CacheEvictionAlgorithm::get_num_blocks_to_evict(size_t decoder_layer_idx) const {
    const size_t L = m_score_manager.get_current_scores_length_in_tokens(decoder_layer_idx);
    if (L == 0)
        return 0;

    // Only start evicting when L exceeds max_cache_size
    if (L <= m_eviction_config.get_max_cache_size())
        return 0;

    const size_t overflow = L - m_eviction_config.get_max_cache_size();
    // Evict full blocks to cover overflow (could add hysteresis)
    return get_num_blocks(overflow);
}

std::size_t CacheEvictionAlgorithm::get_num_evictable_blocks(size_t decoder_layer_idx) const {
    const size_t L = m_score_manager.get_current_scores_length_in_tokens(decoder_layer_idx);
    if (L == 0)
        return 0;
    const size_t total_blocks = get_num_blocks(L);
    const size_t start_blocks = get_num_blocks(m_eviction_config.get_start_size());
    const size_t recent_blocks = get_num_blocks(m_eviction_config.get_recent_size());

    if (total_blocks <= start_blocks + recent_blocks)
        return 0;
    return total_blocks - start_blocks - recent_blocks;
}

CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range(size_t layer_idx) const {
    const size_t L = m_score_manager.get_current_scores_length_in_tokens(layer_idx);
    if (L == 0)
        return CacheEvictionRange::invalid();

    const size_t total_blocks = get_num_blocks(L);
    const size_t start_blocks = get_num_blocks(m_eviction_config.get_start_size());
    const size_t recent_blocks = get_num_blocks(m_eviction_config.get_recent_size());

    if (total_blocks <= start_blocks + recent_blocks)
        return CacheEvictionRange::invalid();

    const size_t begin = start_blocks;
    const size_t end = total_blocks - recent_blocks;
    return CacheEvictionRange(begin, end);
}

std::vector<double> CacheEvictionAlgorithm::get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const {
    const auto range = get_evictable_block_range(decoder_layer_idx);
    if (range == CacheEvictionRange::invalid())
        return {};

    const size_t lbeg = range.first;
    const size_t lend = range.second;
    const size_t blocks = lend - lbeg;

    const auto& vec = m_score_manager.get_scores()[decoder_layer_idx];
    std::vector<double> block_scores(blocks, 0.0);

    for (size_t b = 0; b < blocks; ++b) {
        size_t off = (lbeg + b) * m_block_size;
        for (size_t j = 0; j < m_block_size && (off + j) < vec.size(); ++j) {
            block_scores[b] += vec[off + j];
        }
    }
    return block_scores;
}

std::vector<std::size_t> CacheEvictionAlgorithm::get_indices_of_blocks_to_evict(const std::vector<double>& scores,
                                                                                size_t k) const {
    std::vector<size_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    // Evict the lowest scores
    if (k >= idx.size())
        return idx;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](size_t a, size_t b) {
        return scores[a] < scores[b];
    });
    idx.resize(k);
    // sort ascending (optional)
    std::sort(idx.begin(), idx.end());
    return idx;
}

void CacheEvictionAlgorithm::remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices,
                                                             size_t decoder_layer_idx) {
    m_score_manager.remove_scores(evicted_block_indices, decoder_layer_idx);
}

}  // namespace cache
}  // namespace ov
