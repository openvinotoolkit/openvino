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

// static constexpr size_t BLOCK_SIZE = 8;
// TODO: this value should be equal with prefill chunk size
static constexpr size_t BLOCK_SIZE = 256;

// KV cache tensors for all layers per block
using BlocKVCache = std::vector<std::pair<std::string, ov::SoPtr<ov::ITensor>>>;

struct KVBlock {
    std::vector<uint64_t> token_hashes;
    size_t token_start;
    uint64_t block_hash;
    size_t block_id;
    size_t ref_count;
    bool is_full;

    BlocKVCache block_kv_cache;

    KVBlock() : token_start(0), ref_count(0), is_full(false), block_hash(0) {
        token_hashes.reserve(BLOCK_SIZE);
    }

    /**
     * @brief Add the entire block's token data
     * @param token_hashes Hash values of all tokens in the block
     * @param kv_tensors KV cache data for all tokens in the block
     * @return Whether the addition was successful
     */
    bool add_Block(const std::vector<uint64_t>& token_hashes, const BlocKVCache& kv_tensors) {
        // Check input validity
        if (token_hashes.empty()) {
            return false;
        }

        // Check if the block size exceeds capacity
        if (token_hashes.size() > BLOCK_SIZE) {
            return false;
        }

        // Direct assignment
        this->token_hashes = token_hashes;
        this->block_kv_cache = kv_tensors;
        this->ref_count = token_hashes.size();
        this->is_full = (this->ref_count == BLOCK_SIZE);

        // Compute the block's hash value
        this->block_hash = compute_block_hash(token_hashes);

        return true;
    }

    /**
     * @brief Compute the block's hash value
     * @param token_hashes Hash values of all tokens in the block
     * @return The block's hash value
     *
     * Note: The block hash is derived from the last token hash in the sequence,
     * as each token hash is calculated based on preceding tokens, ensuring
     * a cumulative representation of the block's content.
     */
    uint64_t compute_block_hash(const std::vector<uint64_t>& token_hashes) const {
        // Use the last token hash as the block hash, given token hash is calculated with preceding tokens
        return token_hashes.back();
    }

    void print_block_info(bool verbose) const {
        std::cout << "Block information: " << ref_count << std::endl;
        std::cout << "  Ref Count: " << ref_count << std::endl;
        std::cout << "  Status: " << (is_full ? "Full" : "Not Full") << std::endl;
        std::cout << "  Block index: " << block_id << std::endl;
        std::cout << "  Token start: " << token_start << std::endl;

        if (verbose) {
            std::cout << "  KV cache stored in block: " << std::endl;
        }
        size_t total_size = 0;
        size_t bytes_MB = 1024 * 1024;
        for (const auto& pair : block_kv_cache) {
            const std::string& name = pair.first;
            const ov::SoPtr<ov::ITensor>& tensor = pair.second;

            total_size += tensor->get_byte_size();

            if (!verbose) {
                continue;
            }

            // Print KV cache stored in block verbosely
            std::cout << "Name: " << name << std::endl;
            if (tensor) {
                std::cout << "Tensor Shape: " << tensor->get_shape().to_string() << std::endl;
            } else {
                std::cout << "Tensor is null" << std::endl;
            }
            std::cout << "----------------------------------------" << std::endl;
        }

        std::cout << "  KV cache tensor total size: " << total_size / bytes_MB << " MB" << std::endl;
    }
};

class PrefixCacheManager {
public:
    PrefixCacheManager(size_t max_cache_size = 100) : max_cache_size(max_cache_size) {}

    // Add a block to the cache
    void put_block(const std::shared_ptr<KVBlock>& block);

    // Retrieve a block from the cache by hash
    bool get_block(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block);

    // Print the current status of the cache
    void print_cache_status(bool verbose = false) const;

private:
    size_t max_cache_size;

    // Mapping from hash to KV blocks
    std::unordered_map<uint64_t, std::shared_ptr<KVBlock>> cache_map;

    // LRU list to track the least recently used blocks
    std::list<std::shared_ptr<KVBlock>> lru_list;

    std::mutex mutex;

    // Update the LRU list
    void update_lru(const std::shared_ptr<KVBlock>& block);
};

std::vector<size_t> calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids);

std::unordered_map<std::string, std::string> create_output_to_input_name_mapping(
    const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports);

}  // namespace npuw
}  // namespace ov
