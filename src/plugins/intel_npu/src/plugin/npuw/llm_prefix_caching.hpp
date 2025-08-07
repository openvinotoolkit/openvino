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
    KVBlock(size_t block_size)
        : m_block_size(block_size),
          m_ref_count(0),
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
    size_t m_ref_count;
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

    // Add a block to the cache
    void put_block(const std::shared_ptr<KVBlock>& block, uint64_t prev_block_hash);

    // Retrieve a block from the cache by hash
    bool get_block(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block);

    // Print the current status of the cache
    void print_cache_status(bool verbose = false) const;

private:
    size_t m_max_cache_size;
    std::mutex m_mutex;
    // Mapping from hash to KV blocks
    std::unordered_map<uint64_t, std::shared_ptr<KVBlock>> m_cache_map;
    // LRU list to track the least recently used blocks
    std::list<std::shared_ptr<KVBlock>> m_lru_list;

    // Retrieve a block from the cache by hash without holding a mutex
    std::shared_ptr<KVBlock> get_block_unsafe(uint64_t combined_hash) const;
    void update_lru_unsafe(const std::shared_ptr<KVBlock>& block);
    // Evict the least recently used blocks with no children
    bool evict_lru_block_unsafe();
};

/**
 * @brief Calculate hash values for each token in the input sequence
 *
 * This function computes cumulative hash values for tokens in the input tensor.
 * Each token's hash is calculated based on all preceding tokens, creating a
 * unique fingerprint for each position in the sequence that can be used for
 * prefix caching lookup and storage.
 *
 * @param input_ids Input token tensor containing the token sequence
 * @return Vector of hash values, one for each token position
 */
std::vector<uint64_t> calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids);

/**
 * @brief Create mapping from output tensor names to input tensor names for KV cache
 *
 * This function creates a mapping between the output KV cache tensor names from the
 * prefill model and the corresponding input tensor names that will receive the cached
 * data. This mapping is essential for correctly routing cached KV tensors during
 * prefix cache restoration.
 *
 * @param compiled_model The compiled model containing output port information
 * @param in_ports Map of input port names to their corresponding node outputs
 * @return Mapping from output tensor names to input tensor names
 */
std::unordered_map<std::string, std::string> create_output_to_input_name_mapping(
    const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports);

class LLMInferRequest;

/**
 * @brief Restore cached KV blocks from prefix cache to avoid redundant computation
 *
 * This function attempts to restore previously computed KV cache blocks from the prefix cache
 * based on the input token sequence. It processes tokens in blocks and checks if each block's
 * KV cache data is available in the cache. If found, it copies the cached KV tensors to the
 * corresponding input tensors, significantly reducing computation time for repeated prefixes.
 *
 * @param input_ids Input token tensor containing the token sequence
 * @param block_size Size of each cache block (number of tokens per block)
 * @param prompt_hashes Hash values for each token in the prompt sequence
 * @param input_name_map Mapping from output tensor names to input tensor names for KV cache
 * @param request Reference to LLMInferRequest containing cache manager and model configuration
 * @return Number of tokens successfully restored from cache
 *
 * @note The function processes tokens sequentially in blocks. If a block is not found in cache,
 *       it stops processing and returns the number of tokens restored up to that point.
 */
uint64_t restore_cached_blocks(const ov::SoPtr<ov::ITensor>& input_ids,
                               size_t block_size,
                               const std::vector<uint64_t>& prompt_hashes,
                               const std::unordered_map<std::string, std::string>& input_name_map,
                               LLMInferRequest& request);

/**
 * @brief Store computed KV cache blocks into prefix cache for future reuse
 *
 * This function stores the KV cache tensors computed during inference into the prefix cache
 * organized in blocks. It processes the output tensors from the prefill stage and creates
 * cache blocks that can be retrieved later for the same token sequences, enabling prefix
 * caching optimization for LLM inference.
 *
 * @param chunk_size Size of the current chunk being processed
 * @param block_size Size of each cache block (number of tokens per block)
 * @param prompt_hashes Hash values for each token in the prompt sequence
 * @param token_idx Reference to current token index, updated as blocks are processed
 * @param request Reference to LLMInferRequest containing cache manager and model configuration
 *
 * @note The function only processes chunks that are at least as large as the block size.
 *       It creates cache blocks from the KV output tensors and stores them with proper
 *       parent-child relationships for efficient cache management.
 */
void store_blocks_in_cache(size_t chunk_size,
                           size_t block_size,
                           const std::vector<uint64_t>& prompt_hashes,
                           size_t& token_idx,
                           LLMInferRequest& request);

/**
 * @brief Adjust chunk size to optimize prefix caching block alignment
 *
 * This utility function adjusts the processing chunk size to ensure optimal alignment
 * with prefix caching block boundaries. It helps maximize cache efficiency by ensuring
 * that chunks are processed in sizes that align well with the cached block structure.
 *
 * @param restored_token_num Number of tokens that were successfully restored from cache
 * @param chunk_len Original chunk length to be adjusted
 * @return Adjusted chunk size optimized for prefix caching
 */
size_t adjust_chunk_size_for_prefix_caching(size_t restored_token_num, size_t chunk_len);

}  // namespace npuw
}  // namespace ov
