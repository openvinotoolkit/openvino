// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <regex>
#include <unordered_map>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

/**
 * @brief KV cache data container for storing tensor pairs per layer
 *
 * This type alias defines a container that stores KV cache tensors organized by layer.
 * Each element in the vector represents one layer's KV cache data as a pair of:
 * - string: Tensor name/identifier (e.g., "past_key_values.0.key", "past_key_values.0.values")
 * - ov::SoPtr<ov::ITensor>: Smart pointer to the actual KV cache tensor data
 *
 * The container holds all KV cache tensors for a specific block of tokens,
 * enabling efficient storage and retrieval of cached computation results
 * in the prefix caching system.
 */
using KVData = std::vector<std::pair<std::string, ov::SoPtr<ov::ITensor>>>;

class KVBlock {
public:
    // block_size is number of tokens per block
    explicit KVBlock(size_t block_size)
        : m_block_size(block_size),
          m_is_full(false),
          m_token_start(0),
          m_block_hash(0),
          m_parent_block_hash(0) {
        m_token_hashes.reserve(m_block_size);
    }

    bool is_full() const {
        return m_is_full;
    }

    void set_token_start(size_t token_start) {
        m_token_start = token_start;
    }

    size_t get_token_start() const {
        return m_token_start;
    }

    uint64_t get_block_hash() const {
        return m_block_hash;
    }

    const std::unordered_set<uint64_t>& get_child_block_hashes() const {
        return m_child_block_hashes;
    }

    uint64_t get_parent_block_hash() const {
        return m_parent_block_hash;
    }

    const KVData& get_block_kv_data() const {
        return m_kv_data;
    }

    /**
     * @brief Add the entire block's token data
     * @param token_hashes Hash values of all tokens in the block
     * @param kv_tensors KV cache data for all tokens in the block
     * @return Whether the addition was successful
     */
    bool add_block(const std::vector<uint64_t>& token_hashes, const KVData& kv_tensors);
    void link_blocks(std::shared_ptr<KVBlock> prev_block);
    void unlink_blocks(std::shared_ptr<KVBlock> prev_block);
    void print_block_info(bool verbose) const;

private:
    /**
     * @brief Compute the block's hash value
     * @param token_hashes Hash values of all tokens in the block
     * @return The block's hash value
     *
     * Note: The block hash is derived from the last token hash in the sequence,
     * as each token hash is calculated based on preceding tokens, ensuring
     * a cumulative representation of the block's content.
     */
    uint64_t compute_block_hash(const std::vector<uint64_t>& token_hashes) const;

    size_t m_block_size;
    std::vector<uint64_t> m_token_hashes;
    bool m_is_full;
    size_t m_token_start;
    uint64_t m_block_hash;
    // One block may have multiply child blocks
    std::unordered_set<uint64_t> m_child_block_hashes;
    // One block only has single parent block
    uint64_t m_parent_block_hash;
    KVData m_kv_data;
};

class PrefixCacheManager {
public:
    PrefixCacheManager(size_t max_cache_size) : m_max_cache_size(max_cache_size) {}

    /**
     * @brief Add a block to the cache
     * @param block The KV block to cache
     * @param prev_block_hash Hash of the previous block for linking
     * @return true if the block was successfully added to the cache, false otherwise
     *
     * @note This method does not take ownership of the block. The cache and caller
     *       both hold shared_ptr references. The block's lifetime is managed by
     *       the shared_ptr reference counting mechanism.
     *
     * Reasons for rejection:
     * - Block is not full (!block->is_full())
     * - Block already exists in cache (duplicate)
     * - Previous block is missing and prev_block_hash != 0
     * - Cache is full and no LRU block can be evicted
     */
    bool put_block(const std::shared_ptr<KVBlock>& block, uint64_t prev_block_hash);

    /**
     * @brief Retrieve a block from the cache by hash
     * @param combined_hash Hash value to look up
     * @return Shared pointer to the KVBlock if found, nullptr otherwise
     */
    std::shared_ptr<KVBlock> get_block(uint64_t combined_hash);

    // Print the current status of the cache
    void print_cache_status(bool verbose = false) const;

private:
    size_t m_max_cache_size;
    std::mutex m_mutex;
    // Mapping from hash to KV blocks
    std::unordered_map<uint64_t, std::shared_ptr<KVBlock>> m_cache_map;
    // LRU list for evictable blocks only (leaf nodes with no children, most recent at front)
    std::list<uint64_t> m_evictable_lru_list;
    // Mapping from hash to evictable LRU list iterator for O(1) access
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> m_evictable_lru_iter_map;

    // Retrieve a block from the cache by hash without holding a mutex
    std::shared_ptr<KVBlock> get_block_unsafe(uint64_t combined_hash) const;
    // Update evictable LRU list: mark block as evictable/non-evictable and update its LRU position
    void update_evictable_lru_unsafe(uint64_t block_hash, bool is_evictable);
    // Evict the least recently used blocks with no children
    bool evict_lru_block_unsafe();
};

/**
 * @brief Context structure for prefix caching restoration
 *
 * This structure holds the necessary state and metadata for restoring cached blocks
 * during chunked prefill inference.
 */
struct PrefixCacheRestorationContext {
    std::vector<uint64_t> prompt_hashes;
    size_t token_idx = 0;
    bool restore_prefix_cache = false;
    uint64_t restored_token_num = 0;
    uint64_t remaining_prompts = 0;
};

class LLMInferRequest;

/**
 * @brief Helper class for coordinating prefix caching operations during inference
 *
 * This class encapsulates all prefix caching coordination logic, including:
 * - Hash calculation for token sequences
 * - Cache restoration and storage
 * - Chunk size adjustment for optimal caching
 * - Coordination between inference request and cache manager
 *
 * It serves as a bridge between the LLMInferRequest and PrefixCacheManager,
 * handling all the complex logic required to integrate prefix caching into
 * the inference workflow.
 */
class PrefixCachingHelper {
public:
    /**
     * @brief Construct a PrefixCachingHelper
     * @param request Reference to the LLMInferRequest that owns this helper
     */
    explicit PrefixCachingHelper(LLMInferRequest& request);

    /**
     * @brief Prepare and restore prefix cache before inference
     *
     * This high-level method coordinates the entire cache restoration process.
     */
    PrefixCacheRestorationContext prepare_and_restore(const ov::SoPtr<ov::ITensor>& input_ids,
                                                      uint64_t input_prompt_len);

    /**
     * @brief Store computed KV cache blocks after inference
     */
    void store_computed_blocks(size_t chunk_size, const std::vector<uint64_t>& prompt_hashes, size_t& token_idx);

    /**
     * @brief Print the current status of the prefix cache
     */
    void print_cache_status(bool verbose = false) const;

    /**
     * @brief Populate attention mask for restored prefix cache blocks
     * @param attention_mask Original full attention mask from user input
     * @param attn_mask_in_tensor Attention mask tensor for the current inference chunk
     * @param num_restored_tokens Number of tokens that have been restored from cache
     */
    void populate_attention_mask_for_restored_cache(const ov::SoPtr<ov::ITensor>& attention_mask,
                                                    const ov::SoPtr<ov::ITensor>& attn_mask_in_tensor,
                                                    size_t num_restored_tokens);

private:
    LLMInferRequest& m_request;
    std::shared_ptr<PrefixCacheManager> m_cache_manager;

    // Cached name mapping from output to input ports (initialized once in constructor)
    std::unordered_map<std::string, std::string> m_cached_input_name_map;

    std::vector<uint64_t> calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids);
    void create_name_mapping();

    /**
     * @brief Find cached KV blocks from cache manager
     * @param input_ids Input token tensor
     * @param prompt_hashes Precomputed hash values for all tokens
     * @return Vector of cached KV blocks found in cache
     */
    std::vector<std::shared_ptr<KVBlock>> find_cached_blocks(const ov::SoPtr<ov::ITensor>& input_ids,
                                                             const std::vector<uint64_t>& prompt_hashes);

    /**
     * @brief Copy KV data from cached blocks to prefill request tensors
     * @param cached_blocks Vector of cached KV blocks to copy from
     */
    void copy_cached_kv_data(const std::vector<std::shared_ptr<KVBlock>>& cached_blocks);

    uint64_t restore_blocks(const ov::SoPtr<ov::ITensor>& input_ids, const std::vector<uint64_t>& prompt_hashes);
    void store_blocks(size_t chunk_size, const std::vector<uint64_t>& prompt_hashes, size_t& token_idx);
};

}  // namespace npuw
}  // namespace ov
