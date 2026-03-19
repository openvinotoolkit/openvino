// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_block_manager.hpp"

#include <algorithm>

#include "logging.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

KVCacheBlockManager::KVCacheBlockManager(uint32_t block_size,
                                         uint32_t max_blocks,
                                         const ov::Shape& base_shape,
                                         ov::element::Type elem_type,
                                         const std::string& device,
                                         const std::shared_ptr<const ov::IPlugin>& plugin)
    : block_size_(block_size),
      max_blocks_(max_blocks),
      element_type_(elem_type),
      device_(device),
      plugin_(plugin) {
    // Set block shape: base_shape already contains the full block shape from model
    // For Key:   [batch, num_heads, seq_len, head_dim] e.g., [1, 8, 1024, 128]
    // For Value (transposed): [batch, num_heads, head_dim, seq_len] e.g., [1, 8, 128, 1024]
    // The seq_len dimension already equals block_size, no modification needed
    block_shape_ = base_shape;
    OPENVINO_ASSERT(std::any_of(base_shape.begin(),
                                base_shape.end(),
                                [block_size](size_t dim) {
                                    return dim == block_size;
                                }),
                    "KVCacheBlockManager: base_shape ",
                    base_shape,
                    " does not contain a dimension equal to block_size=",
                    block_size);

    // Initialize block pool (lazy memory allocation)
    blocks_.reserve(max_blocks);
    for (uint32_t i = 0; i < max_blocks; ++i) {
        Block block;
        block.id = i;
        // block.tensor defaults to empty SoPtr (allocate on-demand)
        block.num_tokens = 0;
        block.state = Block::State::FREE;

        blocks_.push_back(std::move(block));
    }

    // Push block IDs to stack in reverse order (so block 0 is on top)
    for (int32_t i = max_blocks - 1; i >= 0; --i) {
        free_block_ids_.push(static_cast<uint32_t>(i));
    }

    LOG_INFO("KVCacheBlockManager initialized: " << "block_size=" << block_size << ", max_blocks=" << max_blocks
                                                 << ", total_capacity=" << (block_size * max_blocks) << " tokens"
                                                 << ", device=" << device);
}

std::optional<uint32_t> KVCacheBlockManager::allocate_block() {
    if (free_block_ids_.empty()) {
        LOG_WARN("KVCacheBlockManager: No free blocks available! " << "All " << max_blocks_ << " blocks are in use.");
        return std::nullopt;
    }

    uint32_t block_id = free_block_ids_.top();
    free_block_ids_.pop();

    auto& block = blocks_[block_id];

    // Allocate actual memory on-demand
    if (!block.tensor) {
        block.tensor = ov::npuw::util::allocMem(element_type_, block_shape_, device_, plugin_);
        LOG_DEBUG("KVCacheBlockManager: Allocated memory for block " << block_id << " (shape=" << block_shape_
                                                                     << ", device=" << device_ << ")");
    }

    // Reset block state
    block.num_tokens = 0;
    block.state = Block::State::ALLOCATED;

    LOG_VERB("KVCacheBlockManager: Allocated block " << block_id
                                                     << " (free blocks remaining: " << free_block_ids_.size() << ")");

    return block_id;
}

ov::SoPtr<ov::ITensor> KVCacheBlockManager::get_block_tensor(uint32_t block_id) {
    validate_block_id(block_id);

    auto& block = blocks_[block_id];

    if (!block.tensor) {
        OPENVINO_THROW("KVCacheBlockManager: Block ",
                       block_id,
                       " has no allocated tensor. "
                       "Call allocate_block() first.");
    }

    return block.tensor;
}

void KVCacheBlockManager::update_block_tokens(uint32_t block_id, uint32_t num_tokens) {
    validate_block_id(block_id);

    if (num_tokens > block_size_) {
        OPENVINO_THROW("KVCacheBlockManager: Cannot set ",
                       num_tokens,
                       " tokens in block ",
                       block_id,
                       " (capacity: ",
                       block_size_,
                       ")");
    }

    auto& block = blocks_[block_id];
    block.num_tokens = num_tokens;

    // Update state
    if (num_tokens >= block_size_) {
        block.state = Block::State::FULL;
    } else if (num_tokens > 0) {
        block.state = Block::State::ALLOCATED;
    }

    LOG_VERB("KVCacheBlockManager: Updated block " << block_id << " tokens: " << num_tokens << "/" << block_size_);
}

uint32_t KVCacheBlockManager::get_block_tokens(uint32_t block_id) const {
    validate_block_id(block_id);
    return blocks_[block_id].num_tokens;
}

std::vector<uint32_t> KVCacheBlockManager::get_allocated_blocks() const {
    std::vector<uint32_t> allocated;
    allocated.reserve(max_blocks_);

    for (const auto& block : blocks_) {
        if (block.state != Block::State::FREE) {
            allocated.push_back(block.id);
        }
    }

    return allocated;
}

void KVCacheBlockManager::clear_all() {
    LOG_DEBUG("KVCacheBlockManager: Clearing all blocks");

    // Free all allocated blocks
    for (auto& block : blocks_) {
        if (block.state != Block::State::FREE) {
            block.num_tokens = 0;
            block.state = Block::State::FREE;
        }
    }

    // Rebuild free stack (push in reverse order so block 0 is on top)
    std::stack<uint32_t> empty;
    free_block_ids_.swap(empty);

    for (int32_t i = max_blocks_ - 1; i >= 0; --i) {
        free_block_ids_.push(static_cast<uint32_t>(i));
    }

    LOG_DEBUG("KVCacheBlockManager: All blocks cleared");
}

void KVCacheBlockManager::validate_block_id(uint32_t block_id) const {
    if (block_id >= max_blocks_) {
        OPENVINO_THROW("KVCacheBlockManager: Invalid block ID ", block_id, " (valid range: 0-", max_blocks_ - 1, ")");
    }
}

}  // namespace npuw
}  // namespace ov
