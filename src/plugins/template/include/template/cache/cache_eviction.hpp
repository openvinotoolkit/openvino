// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/core/type/element_type.hpp"

// Layout contract (critical):
// Each ov::Tensor score passed to register_new_token_scores() must be [T_step, H, L_valid],
// where L_valid is the CURRENT logical token length for the sequence (not max_context_len).
// If your kernel produced [T_step, H, max_ctx], pass a Tensor ROI sliced to L_valid.

using AttentionScoresForCacheOfSubsequence = ov::Tensor;                            // [T_step, H, L_valid]
using AttentionScoresForEachDecoderLayer = std::vector<AttentionScoresForCacheOfSubsequence>;
using AttentionScoresForEachSubsequence = std::map<size_t, AttentionScoresForEachDecoderLayer>;

namespace ov::template_plugin::cache {

/**
 * @brief Represents the mode of per-token score aggregation when determining least important tokens for eviction
 *        from cache
 */
enum class AggregationMode {
    SUM,     /**< Sum attention over steps */
    NORM_SUM /**< Sum divided by lifetime counter (tokens seen) */
};

/**
 * @brief Configuration struct for the cache eviction algorithm.
 */
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
                        "CacheEvictionConfig.max_cache_size must be larger than start_size + recent_size");
        m_evictable_size = m_max_cache_size - m_start_size - m_recent_size;
    }

    std::size_t get_start_size() const { return m_start_size; }
    std::size_t get_recent_size() const { return m_recent_size; }
    std::size_t get_max_cache_size() const { return m_max_cache_size; }
    std::size_t get_evictable_size() const { return m_evictable_size; }

    AggregationMode aggregation_mode = AggregationMode::NORM_SUM;

    // RoPE rotation after eviction (optional integration point)
    bool apply_rotation = false;

    // SnapKV-like window (in query tokens) for max-pooling before aggregating per-step scores
    size_t snapkv_window_size = 8;

private:
    // All three sizes below are in TOKENS and are assumed to be multiples of block_size for this pipeline.
    std::size_t m_start_size = 32;
    std::size_t m_recent_size = 128;
    std::size_t m_max_cache_size = 672;
    std::size_t m_evictable_size = 512;
};

/**
 * @brief Keeps track of the accumulated token scores across model inferences and their lifetime.
 * Storage is per-layer: m_scores[layer][token_idx] and m_cache_counter[layer][token_idx].
 */
class EvictionScoreManager {
public:
    EvictionScoreManager() = default;
    EvictionScoreManager(const EvictionScoreManager& rhs) = default;
    EvictionScoreManager& operator=(const EvictionScoreManager& rhs) = default;

    /**
     * Constructs an EvictionScoreManager.
     * @param block_size Block size of the KV cache (tokens per page).
     * @param num_decoder_layers Number of attention layers.
     * @param max_pool_window_size Query-window size for max-pooling before aggregation (SnapKV).
     * @param aggregation_mode Aggregation mode across steps.
     * @param ignore_first_n_blocks Number of *initial* blocks (from sequence start) whose tokens are never aggregated.
     */
    explicit EvictionScoreManager(size_t block_size,
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

    /**
     * Registers new per-step scores per layer.
     * Each tensor: shape [T_step, H, L_valid] with L_valid == current logical sequence length.
     * @param attention_scores_for_all_decoder_layers size == num_decoder_layers.
     * @param skipped_logical_block_ids logical block IDs (0-based) that were deliberately omitted in the input scores
     *        (e.g., sparse prefill). When non-empty, the provided scores correspond to the logical space with these
     *        blocks removed; they will be "added with gaps" into the tracked vector.
     */
    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers,
                                   const std::set<size_t>& skipped_logical_block_ids);

    /**
     * Removes scores of tokens contained in the given logical blocks for a given layer.
     * (Used after eviction to keep the tracker aligned with remaining tokens.)
     */
    void remove_scores(const std::vector<std::size_t>& evicted_block_indices, size_t decoder_layer_idx);

    /** Adds vectors treating the shorter 'src' as if certain block-sized chunks were skipped. */
    void add_with_skips(std::vector<double>& dst,
                        const std::vector<double>& src,
                        const std::set<size_t>& skipped_logical_block_ids) const;

    size_t get_current_scores_length_in_tokens(size_t layer_idx) const;
    const std::vector<std::vector<double>>& get_scores() const;
    const std::vector<std::vector<size_t>>& get_counters() const;

private:
    std::size_t m_block_size = 0;
    std::size_t m_num_decoder_layers = 0;
    std::vector<std::vector<double>> m_scores;       // [layer][token]
    std::vector<std::vector<size_t>> m_cache_counter; // [layer][token]
    std::size_t m_max_pool_window_size = 1;
    AggregationMode m_aggregation_mode = AggregationMode::NORM_SUM;
    std::size_t m_ignore_first_n_blocks = 0;
};

/**
 * @brief Cache eviction policy based on accumulated attention importance.
 */
class OPENVINO_API CacheEvictionAlgorithm {
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

    explicit CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config,
                                    size_t block_size,
                                    size_t num_decoder_layers,
                                    size_t max_pool_window_size);

    std::size_t get_max_cache_size_after_eviction() const;

    CacheEvictionRange get_evictable_block_range() const;
    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers,
                                   const std::set<size_t>& skipped_logical_block_ids);
    void register_new_token_scores(
        const AttentionScoresForEachDecoderLayer& attention_scores_across_decoder_layers_for_current_sequence);

    /** Returns, per-layer, the logical block IDs to evict now, and updates the tracker accordingly. */
    std::vector<std::set<std::size_t>> evict_logical_blocks();

private:
    static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
        return (a + b - 1) / b;
    }

    std::size_t get_num_blocks(std::size_t num_tokens) const;
    std::size_t get_num_blocks_to_evict(size_t decoder_layer_idx) const;
    std::size_t get_num_evictable_blocks(size_t decoder_layer_idx) const;

    CacheEvictionRange get_evictable_block_range(size_t layer_idx) const;

    std::vector<double> get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const;

    std::vector<std::size_t> get_indices_of_blocks_to_evict(const std::vector<double>& scores_for_each_evictable_block,
                                                            size_t num_blocks_to_evict,
                                                            std::size_t evictable_begin_block) const;

    void remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices,
                                         size_t decoder_layer_idx);

    CacheEvictionConfig m_eviction_config;
    std::size_t m_block_size = 0;
    std::size_t m_num_evicted_tokens = 0; // informational
    std::size_t m_num_decoder_layers = 0;
    EvictionScoreManager m_score_manager;
};

/**
 * @brief Computes RoPE-based rotation parameters after eviction (optional).
 */
class CacheRotationCalculator {
public:
    CacheRotationCalculator(size_t block_size,
                            size_t max_context_length,
                            size_t kv_head_size,
                            double rope_theta = 10000.0f);

    using RotationCoefficientsPerToken = std::vector<std::vector<float>>;  // [BLOCK_SIZE, head_size / 2]

    struct BlockRotationData {
        bool operator==(const BlockRotationData& rhs) const {
            return (logical_block_idx == rhs.logical_block_idx) && (sines == rhs.sines) && (cosines == rhs.cosines);
        }
        size_t logical_block_idx;
        size_t rotation_delta;
        RotationCoefficientsPerToken sines;
        RotationCoefficientsPerToken cosines;
    };

    std::vector<BlockRotationData> get_rotation_data(const std::set<size_t>& evicted_block_logical_indices,
                                                     size_t num_logical_blocks_before_eviction,
                                                     bool deltas_only = true);

    size_t get_head_size() const { return m_head_size; }
    const std::vector<std::vector<float>>& get_sin_lut() const;
    const std::vector<std::vector<float>>& get_cos_lut() const;

private:
    size_t m_block_size;
    size_t m_head_size;
    std::vector<std::vector<float>> m_rope_sin_lut;  // [ max_context_length, head_size / 2]
    std::vector<std::vector<float>> m_rope_cos_lut;  // [ max_context_length, head_size / 2]
};

}  // namespace ov::template_plugin::cache
