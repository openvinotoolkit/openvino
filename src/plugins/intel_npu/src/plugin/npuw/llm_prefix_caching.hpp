// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <regex>
#include <unordered_map>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

static constexpr size_t BLOCK_SIZE = 8;

// KV cache tensors for all layers per token
using KVCachePerToken = std::vector<std::pair<std::string, ov::SoPtr<ov::ITensor>>>;
struct KVBlock {
    std::vector<uint64_t> token_hashes;
    uint64_t block_hash;
    size_t block_id;
    size_t ref_count;
    bool is_full;

    size_t num_layers;
    std::vector<KVCachePerToken> kv_cache_tensors;

    KVBlock() : ref_count(0), is_full(false) {
        token_hashes.resize(BLOCK_SIZE);
        kv_cache_tensors.resize(BLOCK_SIZE);
    }

    bool addToken(uint64_t token_hash, KVCachePerToken current_token_kv) {
        if (is_full) {
            return false;
        }

        token_hashes[ref_count] = token_hash;
        kv_cache_tensors[ref_count] = current_token_kv;
        ref_count ++;

        if (ref_count == BLOCK_SIZE) {
            is_full = true;
        }

        return true;
    }

    uint64_t computeHash() {
        // Use the last token hash as block hash given token hash is calculated with preceding tokens
        return token_hashes.back();
    }
};

class PrefixCacheManager {
public:
    PrefixCacheManager(size_t max_cache_size = 1000)
        : max_cache_size(max_cache_size) {}

    // Put block to cache
    void putBlock(const std::shared_ptr<KVBlock>& block);


    // Get block from cache by hash
    bool getBlock(uint64_t combined_hash,std::shared_ptr<KVBlock>& out_block);

    void printCacheStatus(bool verbose = false);

private:
    size_t max_cache_size;

    // hash to KV blocks
    std::unordered_map<uint64_t, std::shared_ptr<KVBlock>> cache_map;

    std::list<std::shared_ptr<KVBlock>> lru_list;

    std::mutex mutex;

    void updateLRU(const std::shared_ptr<KVBlock>& block);
};


static void printTensorShape(const ov::SoPtr<ov::ITensor>& tensor) {
    if (tensor) {
        std::vector<size_t> shape = tensor->get_shape();
        std::cout << "Tensor Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "Tensor is null" << std::endl;
    }
}

static void printKVCachePerToken(const KVCachePerToken& kv_info) {
    for (const auto& pair : kv_info) {
        const std::string& name = pair.first;
        const ov::SoPtr<ov::ITensor>& tensor = pair.second;

        std::cout << "Name: " << name << std::endl;
        if (tensor) {
            printTensorShape(tensor);
        } else {
            std::cout << "Tensor is null" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }
}

}  // namespace npuw
}  // namespace ov
