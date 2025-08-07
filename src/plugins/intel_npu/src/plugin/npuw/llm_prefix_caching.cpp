// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_prefix_caching.hpp"

namespace ov {
namespace npuw {

void PrefixCacheManager::putBlock(const std::shared_ptr<KVBlock>& block) {
    // Don't cache a non-full block
    if (!block->is_full) {
        return;
    }

    block->block_hash = block->computeHash();

    std::shared_ptr<KVBlock> retrieved_block;
    if (getBlock(block->block_hash, retrieved_block)) {
        // Block has been cached
        updateLRU(retrieved_block);
        return;
    }

    {
        // Add block in cache
        std::lock_guard<std::mutex> lock(mutex);
        // Add block in cache
        block->block_id = cache_map.size();

        if (cache_map.size() >= max_cache_size) {
            // evict LRU block
            auto lru_block = lru_list.back();
            cache_map.erase(lru_block->block_hash);
            lru_list.pop_back();
        }

        cache_map[block->block_hash] = block;
        lru_list.push_front(block);
    }
}

void PrefixCacheManager::updateLRU(const std::shared_ptr<KVBlock>& block) {
    lru_list.remove(block);
    lru_list.push_front(block);
}

bool PrefixCacheManager::getBlock(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache_map.find(combined_hash);
    if (it != cache_map.end()) {
        out_block = it->second;
        updateLRU(out_block);
        return true;
    }

    return false;
}

void PrefixCacheManager::printCacheStatus(bool verbose) {
    std::cout << "Cache Status:" << std::endl;
    std::cout << "Max Cache Size: " << max_cache_size << std::endl;
    std::cout << "Number of Cached Blocks: " << cache_map.size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (const auto& pair : cache_map) {
        uint64_t key = pair.first;
        std::shared_ptr<KVBlock> block = pair.second;

        std::cout << "Block Hash: " << key << std::endl;
        if (block) {
            std::cout << "  Ref Count: " << block->ref_count << std::endl;
            std::cout << "  Status: " << (block->is_full ? "Full" : "Not Full") << std::endl;
            std::cout << "  Block index: " << block->block_id << std::endl;
            if (verbose) {
                std::cout << "  Last token KV info: " << std::endl;
                printKVCachePerToken(block->kv_cache_tensors.back());
            }
            // Add more details as needed
        } else {
            std::cout << "  Block is null" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }
}

}  // namespace npuw
}  // namespace ov
