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

// KV cache tensors for all layers per block
using BlocKVCache = std::vector<std::pair<std::string, ov::SoPtr<ov::ITensor>>>;

class KVBlock {
public:
    size_t m_block_size;
    std::vector<uint64_t> m_token_hashes;
    size_t m_token_start;
    uint64_t m_block_hash;
    size_t m_block_id;
    size_t m_ref_count;
    bool m_is_full;
    BlocKVCache m_block_kv_cache;
    std::chrono::time_point<std::chrono::system_clock> m_timestamp;

    // One block may have multiply children blocks
    std::vector<uint64_t> m_next_block_hashes;
    // One block only has single previous block
    uint64_t m_prev_block_hash;

    KVBlock(size_t block_size)
        : m_block_size(block_size),
          m_token_start(0),
          m_ref_count(0),
          m_is_full(false),
          m_block_hash(0),
          m_prev_block_hash(0) {
        m_token_hashes.reserve(m_block_size);
    }

    /**
     * @brief Add the entire block's token data
     * @param token_hashes Hash values of all tokens in the block
     * @param kv_tensors KV cache data for all tokens in the block
     * @return Whether the addition was successful
     */
    bool add_block(const std::vector<uint64_t>& token_hashes, const BlocKVCache& kv_tensors);

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
};

class PrefixCacheManager {
public:
    PrefixCacheManager(size_t max_cache_size = 100) : m_max_cache_size(max_cache_size) {}

    // Add a block to the cache
    void put_block(const std::shared_ptr<KVBlock>& block, uint64_t prev_block_hash);

    // Retrieve a block from the cache by hash
    bool get_block(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block);

    // Retrieve a block from the cache by hash without holding a mutex
    std::shared_ptr<KVBlock> get_block_unsafe(uint64_t combined_hash);

    // Print the current status of the cache
    void print_cache_status(bool verbose = false) const;

private:
    size_t m_max_cache_size;

    // Mapping from hash to KV blocks
    std::unordered_map<uint64_t, std::shared_ptr<KVBlock>> m_cache_map;

    // Set to track track the least recently used blocks with no children
    struct TimeStampOrder {
        bool operator()(const std::shared_ptr<KVBlock>& lhs, const std::shared_ptr<KVBlock>& rhs) const {
            return lhs->m_timestamp < rhs->m_timestamp;
        }
    };
    std::set<std::shared_ptr<KVBlock>> m_lru_leaf_nodes;

    std::mutex m_mutex;

    // Evict the least recently used blocks with no children
    bool evict_lru_block();
};

std::vector<uint64_t> calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids);

std::unordered_map<std::string, std::string> create_output_to_input_name_mapping(
    const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports);

}  // namespace npuw
}  // namespace ov
