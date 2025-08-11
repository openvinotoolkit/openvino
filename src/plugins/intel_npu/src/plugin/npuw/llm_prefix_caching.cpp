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

    // Direct assignment
    this->token_hashes = token_hashes;
    this->block_kv_cache = kv_tensors;
    this->ref_count = token_hashes.size();
    this->is_full = (this->ref_count == m_block_size);

    // Compute the block's hash value
    this->block_hash = compute_block_hash(token_hashes);

    return true;
}

void KVBlock::link_blocks(std::shared_ptr<KVBlock> prev_block) {
    prev_block->next_block_hashes.push_back(block_hash);
    prev_block_hash = prev_block->block_hash;
}

void KVBlock::unlink_blocks(std::shared_ptr<KVBlock> prev_block) {
    auto curr_block_hash = block_hash;
    auto new_end =
        std::remove(prev_block->next_block_hashes.begin(), prev_block->next_block_hashes.end(), curr_block_hash);
    prev_block->next_block_hashes.erase(new_end, prev_block->next_block_hashes.end());

    prev_block_hash = 0;
}

uint64_t KVBlock::compute_block_hash(const std::vector<uint64_t>& token_hashes) const {
    // Use the last token hash as the block hash, given token hash is calculated with preceding tokens
    return token_hashes.back();
}

void KVBlock::print_block_info(bool verbose) const {
    std::cout << "Block information: " << std::endl;
    std::cout << "  Block size: " << m_block_size << std::endl;
    std::cout << "  Block hash: " << block_hash << std::endl;
    std::cout << "  Ref Count: " << ref_count << std::endl;
    std::cout << "  Status: " << (is_full ? "Full" : "Not Full") << std::endl;
    std::cout << "  Block index: " << block_id << std::endl;
    std::cout << "  Token start: " << token_start << std::endl;

    std::cout << "  Children blocks: " << std::endl;
    for (size_t index = 0; index < next_block_hashes.size(); ++index) {
        std::cout << "    hash [" << index << "]: " << next_block_hashes[index] << std::endl;
    }

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

void PrefixCacheManager::put_block(const std::shared_ptr<KVBlock>& block, uint64_t prev_block_hash) {
    // Do not cache incomplete blocks
    if (!block->is_full) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);

        // Check if the block is already cached
        const auto curr_block = get_block_unsafe(block->block_hash);
        if (curr_block != nullptr) {
            update_lru(curr_block);
            return;
        }

        // Link current block with previous block
        const auto prev_block = get_block_unsafe(prev_block_hash);
        if (prev_block != nullptr) {
            block->link_blocks(prev_block);
        }

        // Add block to cache
        block->block_id = cache_map.size();

        if (cache_map.size() >= max_cache_size) {
            // Evict the least recently used block which does not have any child block
            for (auto lru_it = lru_list.rbegin(); lru_it != lru_list.rend(); ++lru_it) {
                auto lru_block = *lru_it;
                if (lru_block->next_block_hashes.size() != 1) {
                    continue;
                }

                if (lru_block->next_block_hashes.front() != block->block_hash) {
                    continue;
                }
                std::cout << "Cache is full, evict LRU block" << std::endl;
                lru_block->print_block_info(false);

                // Unlink LRU blocks
                const auto lru_prev_block_hash = lru_block->prev_block_hash;
                const auto lru_prev_block = get_block_unsafe(lru_prev_block_hash);
                if (lru_prev_block != nullptr) {
                    lru_block->unlink_blocks(lru_prev_block);
                }

                cache_map.erase(lru_block->block_hash);
                lru_list.pop_back();
                break;  // Exit after evicting one block
            }
        }

        cache_map[block->block_hash] = block;
        lru_list.push_front(block);
    }
}

void PrefixCacheManager::update_lru(const std::shared_ptr<KVBlock>& block) {
    lru_list.remove(block);
    lru_list.push_front(block);
}

bool PrefixCacheManager::get_block(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block) {
    std::lock_guard<std::mutex> lock(mutex);
    out_block = get_block_unsafe(combined_hash);
    if (out_block != nullptr) {
        update_lru(out_block);
        return true;
    }

    return false;
}

std::shared_ptr<KVBlock> PrefixCacheManager::get_block_unsafe(uint64_t combined_hash) {
    auto it = cache_map.find(combined_hash);
    if (it != cache_map.end()) {
        return it->second;
    }

    return nullptr;
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
