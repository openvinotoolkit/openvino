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

    size_t num_layers;
    BlocKVCache block_kv_cache;

    KVBlock() : token_start(0), ref_count(0), is_full(false), block_hash(0) {
        token_hashes.reserve(BLOCK_SIZE);
    }

    /**
     * @brief 一次性添加整个 block 的 token 数据
     * @param token_hashes block 中所有 token 的哈希值
     * @param kv_tensors block 中所有 token 的 KV 缓存数据
     * @return 是否成功添加
     */
    bool addBlock(const std::vector<uint64_t>& token_hashes, const BlocKVCache& kv_tensors) {
        // 检查输入有效性
        if (token_hashes.empty()) {
            return false;
        }

        // 检查是否超出了 block 容量
        if (token_hashes.size() > BLOCK_SIZE) {
            return false;
        }

        // 直接赋值
        this->token_hashes = token_hashes;
        this->block_kv_cache = kv_tensors;
        this->ref_count = token_hashes.size();
        this->is_full = (this->ref_count == BLOCK_SIZE);

        // 计算 block 的哈希值
        this->block_hash = computeBlockHash(token_hashes);

        return true;
    }

    /**
     * @brief 计算 block 的哈希值
     * @param token_hashes block 中所有 token 的哈希值
     * @return block 的哈希值
     */
    uint64_t computeBlockHash(const std::vector<uint64_t>& token_hashes) const {
        // Use the last token hash as block hash given token hash is calculated with preceding tokens
        return token_hashes.back();
    }
};

class PrefixCacheManager {
public:
    PrefixCacheManager(size_t max_cache_size = 1000) : max_cache_size(max_cache_size) {}

    // Put block to cache
    void putBlock(const std::shared_ptr<KVBlock>& block);

    // Get block from cache by hash
    bool getBlock(uint64_t combined_hash, std::shared_ptr<KVBlock>& out_block);

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

static void printBlocKVCache(const BlocKVCache& kv_info) {
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
