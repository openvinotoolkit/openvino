// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_prefix_caching.hpp"

#include "llm_infer_request.hpp"
#include "logging.hpp"

namespace ov {
namespace npuw {

bool KVBlock::add_block(const std::vector<uint64_t>& token_hashes, const BlocKVCache& kv_tensors) {
    // Check input validity
    if (token_hashes.empty()) {
        return false;
    }

    // Check if the block size exceeds capacity
    if (token_hashes.size() > m_block_size) {
        return false;
    }

    m_token_hashes = token_hashes;
    m_block_kv_cache = kv_tensors;
    m_ref_count = token_hashes.size();
    m_is_full = (m_ref_count == m_block_size);

    // Compute the block's hash value
    m_block_hash = compute_block_hash(token_hashes);

    return true;
}

void KVBlock::link_blocks(std::shared_ptr<KVBlock> prev_block) {
    prev_block->m_next_block_hashes.insert(m_block_hash);

    m_prev_block_hash = prev_block->m_block_hash;
}

void KVBlock::unlink_blocks(std::shared_ptr<KVBlock> prev_block) {
    prev_block->m_next_block_hashes.erase(m_block_hash);

    m_prev_block_hash = 0;
}

uint64_t KVBlock::compute_block_hash(const std::vector<uint64_t>& token_hashes) const {
    // Use the last token hash as the block hash, given token hash is calculated with preceding tokens
    return m_token_hashes.back();
}

void KVBlock::print_block_info(bool verbose) const {
    constexpr size_t BYTES_IN_MB = 1024 * 1024;

    std::cout << "Block information: " << std::endl;
    std::cout << "  Block size: " << m_block_size << std::endl;
    std::cout << "  Block hash: " << m_block_hash << std::endl;
    std::cout << "  Ref Count: " << m_ref_count << std::endl;
    std::cout << "  Status: " << (m_is_full ? "Full" : "Not Full") << std::endl;
    std::cout << "  Block index: " << m_block_id << std::endl;
    std::cout << "  Token start: " << m_token_start << std::endl;

    std::cout << "  Children blocks: " << std::endl;
    if (m_next_block_hashes.empty()) {
        std::cout << "    Null" << std::endl;
    } else {
        size_t index = 0;
        for (auto it = m_next_block_hashes.begin(); it != m_next_block_hashes.end(); ++it, ++index) {
            std::cout << "    hash [" << index << "]: " << *it << std::endl;
        }
    }

    if (verbose) {
        std::cout << "  KV cache stored in block: " << std::endl;
    }
    size_t total_size = 0;
    for (const auto& pair : m_block_kv_cache) {
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

    std::cout << "  KV cache tensor total size: " << total_size / BYTES_IN_MB << " MB" << std::endl;
}

void PrefixCacheManager::put_block(const std::shared_ptr<KVBlock>& block, uint64_t prev_block_hash) {
    // Do not cache incomplete blocks
    if (!block->m_is_full) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Check if the block is already cached
        const auto curr_block = get_block_unsafe(block->m_block_hash);
        if (curr_block != nullptr) {
            curr_block->m_timestamp = std::chrono::system_clock::now();
            return;
        }

        // Link current block with previous block
        const auto prev_block = get_block_unsafe(prev_block_hash);
        if (prev_block != nullptr) {
            // Link current block with previous block before evict_lru_block().
            // Consider the scenario where the cache is full and all preceding blocks have a child block.
            // e.g. A -> B -> C -> D we are attempting to insert a new block E into the cache.
            // In this case, Linking blocks first ensures all blocks (A, B, C, D) have child blocks or dependencies,
            // making it impossible to identify an eviction candidate. So that, block E will not be added into the
            // cache.
            block->link_blocks(prev_block);
            // Prev block is not a leaf node
            m_lru_leaf_nodes.erase(prev_block);
        }

        // Add block to cache
        block->m_block_id = m_cache_map.size();

        if (m_cache_map.size() >= m_max_cache_size) {
            if (!evict_lru_block()) {
                // New block is not added into the cache
                if (prev_block != nullptr) {
                    block->unlink_blocks(prev_block);
                }
                return;
            }
        }

        m_cache_map[block->m_block_hash] = block;
        // New added block is a leaf node
        m_lru_leaf_nodes.insert(block);
    }
}

bool PrefixCacheManager::evict_lru_block() {
    if (m_lru_leaf_nodes.empty()) {
        std::cout << "Cache is full but no candidate for eviction" << std::endl;
        return false;
    }

    auto lru_block = *m_lru_leaf_nodes.begin();

    std::cout << "Cache is full, evict LRU block" << std::endl;
    lru_block->print_block_info(false);

    // Unlink LRU blocks
    const auto lru_prev_block_hash = lru_block->m_prev_block_hash;
    const auto lru_prev_block = get_block_unsafe(lru_prev_block_hash);
    if (lru_prev_block != nullptr) {
        lru_block->unlink_blocks(lru_prev_block);
        if (lru_prev_block->m_next_block_hashes.empty()) {
            // LRU prev block is a leaf node
            m_lru_leaf_nodes.insert(lru_prev_block);
        }
    }

    m_cache_map.erase(lru_block->m_block_hash);
    m_lru_leaf_nodes.erase(m_lru_leaf_nodes.begin());

    return true;
}

bool PrefixCacheManager::get_block(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block) {
    std::lock_guard<std::mutex> lock(m_mutex);
    out_block = get_block_unsafe(combined_hash);
    if (out_block != nullptr) {
        out_block->m_timestamp = std::chrono::system_clock::now();
        return true;
    }

    return false;
}

std::shared_ptr<KVBlock> PrefixCacheManager::get_block_unsafe(uint64_t combined_hash) {
    auto it = m_cache_map.find(combined_hash);
    if (it != m_cache_map.end()) {
        return it->second;
    }

    return nullptr;
}

void PrefixCacheManager::print_cache_status(bool verbose) const {
    std::cout << "Cache Status:" << std::endl;
    std::cout << "Max Cache Size: " << m_max_cache_size << " blocks" << std::endl;
    std::cout << "Number of Cached Blocks: " << m_cache_map.size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Print information of all blocks in cache
    for (const auto& pair : m_cache_map) {
        uint64_t key = pair.first;
        std::shared_ptr<KVBlock> block = pair.second;

        std::cout << "Key Hash: " << key << std::endl;
        if (block) {
            block->print_block_info(verbose);
        } else {
            std::cout << "  Block is null" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }
}

std::vector<uint64_t> calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids) {
    const char* data = reinterpret_cast<const char*>(input_ids->data());
    const auto data_elem_size = input_ids->get_element_type().size();
    size_t total_size = input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM];

    std::vector<uint64_t> prompt_hashes(total_size);

    uint64_t prefix_hash = 0;
    for (size_t i = 0; i < total_size; ++i) {
        const char* token_data = reinterpret_cast<const char*>(input_ids->data()) + i * data_elem_size;
        uint64_t token_hash = std::hash<std::string_view>{}(std::string_view(token_data, data_elem_size));
        prefix_hash = prefix_hash * 31 + token_hash;
        prompt_hashes[i] = prefix_hash;
    }

    return prompt_hashes;
}

std::unordered_map<std::string, std::string> create_output_to_input_name_mapping(
    const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports) {
    std::unordered_map<std::string, std::string> input_name_map;
    for (std::size_t i = kStartOutputKVCacheLayers; i < compiled_model->outputs().size(); ++i) {
        const auto& output_name = compiled_model->outputs()[i].get_any_name();
        std::string input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
        if (in_ports.find(input_name) == in_ports.end()) {
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }

        input_name_map[output_name] = input_name;
    }

    return input_name_map;
}

}  // namespace npuw
}  // namespace ov
