// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov { namespace cache {

using AttentionScoresForCacheOfSubsequence = ov::Tensor;
using AttentionScoresForEachDecoderLayer = std::vector<AttentionScoresForCacheOfSubsequence>;

/**
 * @brief Represents how we aggregate per-token scores across steps.
 */
enum class AggregationMode {
    SUM,
    NORM_SUM // divide by lifetime
};

class CacheEvictionConfig {
public:
    CacheEvictionConfig() = default;

    CacheEvictionConfig(size_t start_size,
                        size_t recent_size,
                        size_t max_cache_size,
                        AggregationMode aggregation_mode_,
                        bool apply_rotation_ = false,
                        size_t snapkv_window_size_ = 8)
        : aggregation_mode(aggregation_mode_),
          apply_rotation(apply_rotation_),
          snapkv_window_size(snapkv_window_size_),
          m_start_size(start_size),
          m_recent_size(recent_size),
          m_max_cache_size(max_cache_size) {
        OPENVINO_ASSERT(start_size, "CacheEvictionConfig.start_size must be non-zero");
        OPENVINO_ASSERT(recent_size, "CacheEvictionConfig.recent_size must be non-zero");
        OPENVINO_ASSERT(max_cache_size, "CacheEvictionConfig.max_cache_size must be non-zero");
        OPENVINO_ASSERT(snapkv_window_size, "CacheEvictionConfig.snapkv_window_size must be non-zero");
        OPENVINO_ASSERT(max_cache_size > (start_size + recent_size),
                        "max_cache_size must be larger than start_size + recent_size");
        m_evictable_size = m_max_cache_size - m_start_size - m_recent_size;
    }

    std::size_t get_start_size() const { return m_start_size; }
    std::size_t get_recent_size() const { return m_recent_size; }
    std::size_t get_max_cache_size() const { return m_max_cache_size; }
    std::size_t get_evictable_size() const { return m_evictable_size; }

    AggregationMode aggregation_mode = AggregationMode::NORM_SUM;
    bool apply_rotation = false;
    size_t snapkv_window_size = 8;

private:
    std::size_t m_start_size = 32;
    std::size_t m_recent_size = 128;
    std::size_t m_max_cache_size = 672;
    std::size_t m_evictable_size = 512;
};

/**
 * @brief Tracks accumulated token scores per layer and lifetimes.
 * Internal representation: per-layer vectors sized by current token length.
 */
class EvictionScoreManager {
public:
    EvictionScoreManager() = default;
    EvictionScoreManager(size_t block_size,
                         size_t num_decoder_layers,
                         size_t max_pool_window_size,
                         AggregationMode aggregation_mode,
                         size_t ignore_first_n_blocks = 0)
        : m_block_size(block_size),
          m_num_decoder_layers(num_decoder_layers),
          m_scores(num_decoder_layers),
          m_cache_counter(num_decoder_layers),
          m_max_pool_window_size(max_pool_window_size),
          m_aggregation_mode(aggregation_mode),
          m_ignore_first_n_blocks(ignore_first_n_blocks) {}

    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers,
                                   const std::set<size_t>& skipped_logical_block_ids);

    void remove_scores(const std::vector<std::size_t>& evicted_block_indices, size_t decoder_layer_idx);

    void add_with_skips(std::vector<double>& dst,
                        const std::vector<double>& src,
                        const std::set<size_t>& skipped_logical_block_ids) const;

    size_t get_current_scores_length_in_tokens(size_t layer_idx) const {
        return m_scores[layer_idx].size();
    }

    const std::vector<std::vector<double>>& get_scores() const { return m_scores; }
    const std::vector<std::vector<size_t>>& get_counters() const { return m_cache_counter; }

private:
    std::size_t m_block_size = 0;
    std::size_t m_num_decoder_layers = 0;
    std::vector<std::vector<double>> m_scores;         // [layer][token]
    std::vector<std::vector<size_t>> m_cache_counter;  // [layer][token]
    std::size_t m_max_pool_window_size = 1;
    AggregationMode m_aggregation_mode = AggregationMode::SUM;
    std::size_t m_ignore_first_n_blocks = 0;
};

/**
 * @brief Main eviction policy.
 */
class CacheEvictionAlgorithm {
public:
    class CacheEvictionRange : public std::pair<std::size_t, std::size_t> {
    public:
        CacheEvictionRange(std::size_t begin, std::size_t end) : std::pair<std::size_t, std::size_t>(begin, end) {}
        static const CacheEvictionRange& invalid() {
            static CacheEvictionRange inv(0, 0);
            return inv;
        }
    };

    CacheEvictionAlgorithm() = default;

    CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config,
                           size_t block_size,
                           size_t num_decoder_layers,
                           size_t max_pool_window_size);

    std::size_t get_max_cache_size_after_eviction() const;

    CacheEvictionRange get_evictable_block_range() const; // default: layer 0

    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& per_layer_scores,
                                   const std::set<size_t>& skipped_logical_block_ids);

    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& per_layer_scores_only);

    std::vector<std::set<std::size_t>> evict_logical_blocks(); // per-layer result (same set across layers for now)

private:
    std::size_t get_num_blocks(std::size_t num_tokens) const;
    std::size_t get_num_blocks_to_evict(size_t decoder_layer_idx) const;
    std::size_t get_num_evictable_blocks(size_t decoder_layer_idx) const;

    CacheEvictionRange get_evictable_block_range(size_t layer_idx) const;

    std::vector<double> get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const;

    std::vector<std::size_t> get_indices_of_blocks_to_evict(const std::vector<double>& scores_for_each_evictable_block,
                                                            size_t num_blocks_to_evict) const;

    void remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices,
                                         size_t decoder_layer_idx);

    CacheEvictionConfig m_eviction_config;
    std::size_t m_block_size = 0;
    std::size_t m_num_evicted_tokens = 0;
    std::size_t m_num_decoder_layers = 0;
    EvictionScoreManager m_score_manager;
};

/**
 * @brief Optional RoPE rotation calculator (deltas only). Stubbed for now.
 */
class CacheRotationCalculator {
public:
    CacheRotationCalculator(size_t block_size,
                            size_t max_context_length,
                            size_t kv_head_size,
                            double rope_theta = 10000.0f)
        : m_block_size(block_size), m_head_size(kv_head_size) {}

    struct BlockRotationData {
        size_t logical_block_idx = 0;
        size_t rotation_delta = 0;
    };

    std::vector<BlockRotationData> get_rotation_data(const std::set<size_t>& evicted_block_logical_indices,
                                                     size_t num_logical_blocks_before_eviction,
                                                     bool deltas_only = true) {
        (void)evicted_block_logical_indices;
        (void)num_logical_blocks_before_eviction;
        (void)deltas_only;
        return {};
    }

    size_t get_head_size() const { return m_head_size; }

private:
    size_t m_block_size = 0;
    size_t m_head_size = 0;
};

}} // namespace ov::cache
