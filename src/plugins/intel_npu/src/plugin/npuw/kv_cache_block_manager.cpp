// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_block_manager.hpp"

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
      block_shape_(base_shape),
      device_(device),
      plugin_(plugin) {
    OPENVINO_ASSERT(block_size_ > 0 && max_blocks_ > 0,
                    "KVCacheBlockManager: block_size and max_blocks must be > 0, got block_size=",
                    block_size_,
                    " max_blocks=",
                    max_blocks_);
    OPENVINO_ASSERT(device_ == "CPU" || plugin_ != nullptr,
                    "KVCacheBlockManager: plugin must be non-null for non-CPU device '",
                    device_,
                    "'");
    // Check that the sequence dimension (dim 2 for K/non-transposed-V, dim 3 for transposed V)
    // equals block_size.
    OPENVINO_ASSERT(base_shape.size() == 4 && (base_shape[2] == block_size || base_shape[3] == block_size),
                    "KVCacheBlockManager: base_shape ",
                    base_shape,
                    " does not have block_size=",
                    block_size,
                    " in sequence dimension (expected at dim 2 or 3)");

    // Initialize block pool (tensors allocated on-demand, not here)
    blocks_.reserve(max_blocks);
    for (uint32_t i = 0; i < max_blocks; ++i) {
        blocks_.push_back({});
    }

    LOG_INFO("KVCacheBlockManager initialized: " << "block_size=" << block_size << ", max_blocks=" << max_blocks
                                                 << ", total_capacity=" << (block_size * max_blocks) << " tokens"
                                                 << ", device=" << device);
}

std::optional<uint32_t> KVCacheBlockManager::allocate_block() {
    const auto free_id = first_free_block_id();
    if (!free_id) {
        LOG_WARN("KVCacheBlockManager: No free blocks available! " << "All " << max_blocks_ << " blocks are in use.");
        return std::nullopt;
    }

    const uint32_t block_id = *free_id;

    auto& block = blocks_[block_id];

    // Allocate actual memory on-demand
    if (!block.tensor) {
        block.tensor = ov::npuw::util::allocMem(element_type_, block_shape_, device_, plugin_);
        LOG_DEBUG("KVCacheBlockManager: Allocated memory for block " << block_id << " (shape=" << block_shape_
                                                                     << ", device=" << device_ << ")");
    }

    // Reset block state
    block.num_tokens = 0;
    block.is_allocated = true;
    block.is_shared = false;

    LOG_VERB("KVCacheBlockManager: Allocated block " << block_id << " (free blocks remaining: " << num_free_blocks()
                                                     << ")");

    return block_id;
}

uint32_t KVCacheBlockManager::allocate_block_with_tensor(ov::SoPtr<ov::ITensor> tensor, uint32_t num_tokens) {
    OPENVINO_ASSERT(tensor, "KVCacheBlockManager: cannot inject a null tensor");
    OPENVINO_ASSERT(num_tokens <= block_size_,
                    "KVCacheBlockManager: cannot inject ",
                    num_tokens,
                    " tokens (capacity: ",
                    block_size_,
                    ")");
    const auto free_id = first_free_block_id();
    OPENVINO_ASSERT(free_id.has_value(), "KVCacheBlockManager: no free blocks available for prefix cache restore");

    const uint32_t block_id = *free_id;

    auto& block = blocks_[block_id];
    block.tensor = std::move(tensor);
    block.num_tokens = num_tokens;
    block.is_allocated = true;
    block.is_shared = true;

    LOG_VERB("KVCacheBlockManager: Injected cached tensor into block " << block_id << " (" << num_tokens << " tokens)");
    return block_id;
}

void KVCacheBlockManager::mark_block_shared(uint32_t block_id) {
    validate_block_id(block_id);
    OPENVINO_ASSERT(blocks_[block_id].is_allocated, "KVCacheBlockManager: cannot share free block ", block_id);
    blocks_[block_id].is_shared = true;
}

bool KVCacheBlockManager::make_block_private(uint32_t block_id) {
    validate_block_id(block_id);
    auto& block = blocks_[block_id];
    OPENVINO_ASSERT(block.is_allocated, "KVCacheBlockManager: cannot make free block ", block_id, " private");
    if (!block.is_shared) {
        return false;
    }
    auto copy = ov::npuw::util::allocMem(element_type_, block_shape_, device_, plugin_);
    block.tensor->copy_to(copy._ptr);
    block.tensor = std::move(copy);
    block.is_shared = false;
    return true;
}

ov::SoPtr<ov::ITensor> KVCacheBlockManager::get_block_tensor(uint32_t block_id) const {
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

    if (!blocks_[block_id].is_allocated) {
        OPENVINO_THROW("KVCacheBlockManager: Cannot update tokens on free block ",
                       block_id,
                       ". Call allocate_block() first.");
    }

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

    LOG_VERB("KVCacheBlockManager: Updated block " << block_id << " tokens: " << num_tokens << "/" << block_size_);
}

uint32_t KVCacheBlockManager::get_block_tokens(uint32_t block_id) const {
    validate_block_id(block_id);
    return blocks_[block_id].num_tokens;
}

std::vector<uint32_t> KVCacheBlockManager::get_allocated_blocks() const {
    std::vector<uint32_t> allocated;
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].is_allocated) {
            allocated.push_back(i);
        }
    }
    return allocated;
}

uint32_t KVCacheBlockManager::num_free_blocks() const {
    uint32_t free_blocks = 0;
    for (const auto& block : blocks_) {
        free_blocks += block.is_allocated ? 0u : 1u;
    }
    return free_blocks;
}

void KVCacheBlockManager::release(uint32_t keep_warm_count) {
    LOG_DEBUG("KVCacheBlockManager: Resetting blocks (keep_warm_count=" << keep_warm_count << ")");

    // Iterate allocated blocks in ascending block-ID order.
    // The first keep_warm_count retain their device tensors (warm reuse next round).
    // The remainder have tensors dropped to reduce RSS.
    uint32_t warm_seen = 0;
    for (auto& block : blocks_) {
        if (!block.is_allocated) {
            continue;
        }
        block.is_allocated = false;
        block.num_tokens = 0;
        if (block.is_shared) {
            // The prefix cache still references this tensor, so a later conversation must not
            // reuse it as scratch memory. Only the pool's reference is dropped here.
            block.tensor = {};
            block.is_shared = false;
            continue;
        }
        if (warm_seen >= keep_warm_count) {
            block.tensor = {};  // Drop SoPtr — returns device memory to allocator
        }
        ++warm_seen;
    }

    if (warm_seen > keep_warm_count) {
        LOG_DEBUG("KVCacheBlockManager: Released " << (warm_seen - keep_warm_count) << " block tensor(s), kept "
                                                   << keep_warm_count << " warm");
    }
}

void KVCacheBlockManager::clear_all() {
    release(0);
}

std::optional<uint32_t> KVCacheBlockManager::first_free_block_id() const {
    // Smallest free ID first: release() keeps the lowest IDs warm, so those tensors are reused first.
    for (uint32_t i = 0; i < max_blocks_; ++i) {
        if (!blocks_[i].is_allocated) {
            return i;
        }
    }
    return std::nullopt;
}

void KVCacheBlockManager::truncate_allocated(uint32_t keep_count) {
    const auto allocated = get_allocated_blocks();
    OPENVINO_ASSERT(keep_count <= allocated.size(),
                    "KVCacheBlockManager: cannot truncate to ",
                    keep_count,
                    " blocks, only ",
                    allocated.size(),
                    " are allocated");

    // Deallocate the suffix, keeping device tensors warm for quick reuse. Shared tensors are
    // still referenced by the prefix cache, so they are dropped instead.
    for (size_t i = keep_count; i < allocated.size(); ++i) {
        auto& block = blocks_[allocated[i]];
        block.is_allocated = false;
        block.num_tokens = 0;
        if (block.is_shared) {
            block.tensor = {};
            block.is_shared = false;
        }
    }

    LOG_DEBUG("KVCacheBlockManager: truncated to " << keep_count << " blocks, " << (allocated.size() - keep_count)
                                                   << " suffix block(s) freed");
}

void KVCacheBlockManager::validate_block_id(uint32_t block_id) const {
    if (block_id >= max_blocks_) {
        OPENVINO_THROW("KVCacheBlockManager: Invalid block ID ", block_id, " (valid range: 0-", max_blocks_ - 1, ")");
    }
}

}  // namespace npuw
}  // namespace ov
