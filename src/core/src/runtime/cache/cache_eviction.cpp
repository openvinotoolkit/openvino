// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/runtime/cache/cache_eviction.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ov::cache {

// ============ EvictionScoreManager ============

void EvictionScoreManager::register_new_token_scores(
    const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers,
    const std::set<size_t>& skipped_logical_block_ids) {
    // TODO: Implement registration logic
}

void EvictionScoreManager::remove_scores(const std::vector<std::size_t>& evicted_block_indices,
                                         size_t decoder_layer_idx) {
    // TODO: Implement logic to remove tracked scores
}

void EvictionScoreManager::add_with_skips(std::vector<double>& dst,
                                          const std::vector<double>& src,
                                          const std::set<size_t>& skipped_logical_block_ids) const {
    // TODO: Implement logic to add vectors with skipped indices
}

size_t EvictionScoreManager::get_current_scores_length_in_tokens(size_t layer_idx) const {
    // TODO: Implement retrieval logic
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
                      eviction_config.get_start_size() / block_size) {
    // Init logic
}

std::size_t CacheEvictionAlgorithm::get_max_cache_size_after_eviction() const {
    return m_eviction_config.get_max_cache_size() + m_block_size - 1;
}

CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range() const {
    // Just use layer 0 for default
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
    // TODO: Add real eviction logic here
    return result;
}

std::size_t CacheEvictionAlgorithm::get_num_blocks(std::size_t num_tokens) const {
    return (num_tokens + m_block_size - 1) / m_block_size;
}

std::size_t CacheEvictionAlgorithm::get_num_blocks_to_evict(size_t decoder_layer_idx) const {
    // TODO: Implement
    return 0;
}

std::size_t CacheEvictionAlgorithm::get_num_evictable_blocks(size_t decoder_layer_idx) const {
    // TODO
    return 0;
}

CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range(size_t layer_idx) const {
    // TODO: Compute using current token count in layer `layer_idx`
    return CacheEvictionRange::invalid();
}

std::vector<double> CacheEvictionAlgorithm::get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const {
    // TODO
    return {};
}

std::vector<std::size_t> CacheEvictionAlgorithm::get_indices_of_blocks_to_evict(
    const std::vector<double>& scores_for_each_evictable_block,
    size_t num_blocks_to_evict) const {
    // TODO
    return {};
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
    // Precompute LUT
    for (size_t pos = 0; pos < max_context_length; ++pos) {
        for (size_t i = 0; i < kv_head_size / 2; ++i) {
            float theta = static_cast<float>(1.0 / std::pow(rope_theta, static_cast<float>(i) / (kv_head_size / 2)));
            float angle = pos * theta;
            m_rope_sin_lut[pos][i] = std::sin(angle);
            m_rope_cos_lut[pos][i] = std::cos(angle);
        }
    }
}

std::vector<CacheRotationCalculator::BlockRotationData> CacheRotationCalculator::get_rotation_data(
    const std::set<size_t>& evicted_block_logical_indices,
    size_t num_logical_blocks_before_eviction,
    bool deltas_only) {
    std::vector<BlockRotationData> result;
    // TODO: Implement real rotation logic
    return result;
}

const std::vector<std::vector<float>>& CacheRotationCalculator::get_sin_lut() const {
    return m_rope_sin_lut;
}

const std::vector<std::vector<float>>& CacheRotationCalculator::get_cos_lut() const {
    return m_rope_cos_lut;
}

}  // namespace ov::cache
