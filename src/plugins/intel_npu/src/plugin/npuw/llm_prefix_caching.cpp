// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_prefix_caching.hpp"

#include "llm_infer_request.hpp"
#include "logging.hpp"

namespace ov {
namespace npuw {

bool KVBlock::add_Block(const std::vector<uint64_t>& token_hashes, const BlocKVCache& kv_tensors) {
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

uint64_t KVBlock::compute_block_hash(const std::vector<uint64_t>& token_hashes) const {
    // Use the last token hash as the block hash, given token hash is calculated with preceding tokens
    return token_hashes.back();
}

void KVBlock::print_block_info(bool verbose) const {
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

void PrefixCacheManager::put_block(const std::shared_ptr<KVBlock>& block) {
    // Do not cache incomplete blocks
    if (!block->is_full) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    // Check if the block is already cached
    auto it = cache_map.find(block->block_hash);
    if (it != cache_map.end()) {
        update_lru(it->second);
        return;
    }

    // Add block to cache
    block->block_id = cache_map.size();

    if (cache_map.size() >= max_cache_size) {
        // Evict the least recently used block
        auto lru_block = lru_list.back();
        cache_map.erase(lru_block->block_hash);
        lru_list.pop_back();
    }

    cache_map[block->block_hash] = block;
    lru_list.push_front(block);
}

void PrefixCacheManager::update_lru(const std::shared_ptr<KVBlock>& block) {
    lru_list.remove(block);
    lru_list.push_front(block);
}

bool PrefixCacheManager::get_block(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache_map.find(combined_hash);
    if (it != cache_map.end()) {
        out_block = it->second;
        update_lru(out_block);
        return true;
    }

    return false;
}

void PrefixCacheManager::print_cache_status(bool verbose) const {
    std::cout << "Cache Status:" << std::endl;
    std::cout << "Max Cache Size: " << max_cache_size << " blocks" << std::endl;
    std::cout << "Number of Cached Blocks: " << cache_map.size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Print information of all blocks in cache
    for (const auto& pair : cache_map) {
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

std::vector<size_t> calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids) {
    const char* data = reinterpret_cast<const char*>(input_ids->data());
    const auto data_elem_size = input_ids->get_element_type().size();
    size_t total_size = input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM];

    std::vector<size_t> prompt_hashes(total_size);

    size_t prefix_hash = 0;
    for (size_t i = 0; i < total_size; ++i) {
        const char* token_data = reinterpret_cast<const char*>(input_ids->data()) + i * data_elem_size;
        size_t token_hash = std::hash<std::string_view>{}(std::string_view(token_data, data_elem_size));
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
