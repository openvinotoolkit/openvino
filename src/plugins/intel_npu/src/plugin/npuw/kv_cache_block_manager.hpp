// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace test {
namespace npuw {
struct KVCacheBlockManagerTestAccess;
}  // namespace npuw
}  // namespace test
}  // namespace ov

namespace ov {
namespace npuw {

class LLMBlockKVCacheStrategy;

/**
 * @brief Block-based KV Cache Manager
 *
 * Manages KV cache memory using fixed-size blocks instead of a single continuous buffer.
 * This approach significantly reduces memory waste when actual prompt length is much smaller
 * than the configured max_prompt_size.
 *
 * Key benefits:
 * - Memory efficiency: Only allocate blocks as needed
 * - Flexible allocation: Adapt to variable-length prompts without pre-allocating max buffer
 *
 * Example usage:
 *   KVCacheBlockManager manager(512, 16, shape, type, "NPU", plugin);
 *   auto block_id = manager.allocate_block();
 *   auto tensor = manager.get_block_tensor(block_id.value());
 *   manager.update_block_tokens(block_id.value(), 256);
 *   manager.clear_all();
 */
class KVCacheBlockManager {
public:
    /**
     * @brief Construct a new KV Cache Block Manager
     *
     * @param block_size Number of tokens per block
     * @param max_blocks Maximum number of blocks in the pool
     * @param base_shape Base shape for block tensors [batch, num_heads, seq_len, head_dim]
     * @param elem_type Element type (e.g., fp16, fp32)
     * @param device Target device for memory allocation ("NPU", "CPU")
     * @param plugin Plugin instance for memory allocation
     */
    KVCacheBlockManager(uint32_t block_size,
                        uint32_t max_blocks,
                        const ov::Shape& base_shape,
                        ov::element::Type elem_type,
                        const std::string& device,
                        const std::shared_ptr<const ov::IPlugin>& plugin);

    ~KVCacheBlockManager() = default;

    // Disable copy
    KVCacheBlockManager(const KVCacheBlockManager&) = delete;
    KVCacheBlockManager& operator=(const KVCacheBlockManager&) = delete;

    // Allow move
    KVCacheBlockManager(KVCacheBlockManager&&) = default;
    KVCacheBlockManager& operator=(KVCacheBlockManager&&) = default;

    /**
     * @brief Allocate a new block from the free pool
     *
     * @return Block ID if successful, std::nullopt if no free blocks available
     */
    std::optional<uint32_t> allocate_block();

    /**
     * @brief Get the tensor associated with a block
     *
     * @param block_id Block ID
     * @return Tensor for the block
     */
    ov::SoPtr<ov::ITensor> get_block_tensor(uint32_t block_id) const;

    /**
     * @brief Update the number of tokens stored in a block
     *
     * @param block_id Block ID
     * @param num_tokens New token count (must be <= block_size)
     */
    void update_block_tokens(uint32_t block_id, uint32_t num_tokens);

    /**
     * @brief Get the number of tokens in a block
     *
     * @param block_id Block ID
     * @return Number of tokens
     */
    uint32_t get_block_tokens(uint32_t block_id) const;

    /**
     * @brief Get all currently allocated block IDs in ascending order
     *
     * Blocks are always handed out smallest ID first, so index i is the block holding tokens
     * [i * block_size, (i + 1) * block_size).
     *
     * @return Vector of block IDs
     */
    std::vector<uint32_t> get_allocated_blocks() const;

    /**
     * @brief Release blocks to FREE state, selectively dropping device memory.
     *
     * The first `keep_warm_count` allocated blocks (by block ID, ascending) retain
     * their device tensors so they can be reused in the next conversation without
     * re-allocating device memory.  Blocks beyond that count have their tensors
     * dropped to reduce RSS.
     *
     * @param keep_warm_count Number of blocks whose device tensors should be kept.
     *                        Pass 0 (default) to release all device memory.
     *
     * Note: for NPU-backed tensors, re-allocation adds latency to the first prefill
     * chunk of a new conversation.  Keeping `ceil(prompt_len / block_size)` blocks
     * warm avoids that latency for the common case.
     */
    void release(uint32_t keep_warm_count = 0);

    /**
     * @brief Release all blocks and drop all device tensor memory.
     *
     * Equivalent to release(0).  Kept for compatibility.
     */
    void clear_all();

    /**
     * @brief Truncate the allocated block list to its first keep_count blocks.
     *
     * Used by continuous prefill to discard the suffix of a previous turn while
     * retaining the shared prefix. Retained blocks keep their IDs, order, token
     * counts and tensors. Suffix blocks are deallocated with token counts zeroed
     * and their device tensors kept warm, and subsequent allocations return the
     * freed IDs in ascending order so logical append order is preserved.
     *
     * @param keep_count Number of leading allocated blocks to retain. Must not
     *                   exceed the current allocated count.
     */
    void truncate_allocated(uint32_t keep_count);

    /**
     * @brief Get block size (tokens per block)
     */
    uint32_t get_block_size() const {
        return block_size_;
    }

    /**
     * @brief Get maximum number of blocks
     */
    uint32_t get_max_blocks() const {
        return max_blocks_;
    }

    /**
     * @brief Get number of currently free (unallocated) blocks
     */
    uint32_t num_free_blocks() const;

private:
    friend class LLMBlockKVCacheStrategy;
    friend struct ov::test::npuw::KVCacheBlockManagerTestAccess;

    /**
     * @brief Make a free block active by injecting an existing tensor, without allocating memory.
     *
     * Used to restore a prefix-cached block. The tensor is marked shared, so release() and
     * truncate_allocated() drop it instead of keeping it warm for reuse.
     * Must be called in ascending token order: blocks are handed out smallest ID first, which is
     * what keeps block ID order equal to token order.
     *
     * @return ID of the block that now holds the tensor
     */
    uint32_t allocate_block_with_tensor(ov::SoPtr<ov::ITensor> tensor, uint32_t num_tokens);

    /**
     * @brief Mark an allocated block as also referenced by the prefix cache.
     *
     * A shared block must never be rewritten by a later conversation, so its tensor is dropped
     * from the pool (not kept warm) when the block is released.
     */
    void mark_block_shared(uint32_t block_id);

    /**
     * @brief Give an allocated shared block a private copy of its tensor.
     *
     * Needed before rewriting a block in place, so the prefix cache keeps the original data.
     * The block ID and token count are unchanged. A no-op for a block that is not shared.
     *
     * @return true if the tensor was replaced, so any binding to the old tensor is stale
     */
    bool make_block_private(uint32_t block_id);

    /**
     * @brief Represents a single block of KV cache memory
     */
    struct Block {
        ov::SoPtr<ov::ITensor> tensor;  ///< Block memory tensor (allocated on-demand)
        uint32_t num_tokens = 0;        ///< Number of tokens stored in this block
        bool is_allocated = false;      ///< True when block is in use; false = free
        bool is_shared = false;         ///< True when the tensor is also referenced by the prefix cache
    };

    uint32_t block_size_;        ///< Number of tokens per block
    uint32_t max_blocks_;        ///< Maximum blocks in pool
    std::vector<Block> blocks_;  ///< All blocks (free + allocated)

    ov::element::Type element_type_;             ///< Element type for tensors
    ov::Shape block_shape_;                      ///< Shape for block tensors
    std::string device_;                         ///< Target device
    std::shared_ptr<const ov::IPlugin> plugin_;  ///< Plugin for memory allocation

    /**
     * @brief Smallest block ID that is not allocated, or std::nullopt if the pool is exhausted.
     */
    std::optional<uint32_t> first_free_block_id() const;

    /**
     * @brief Validate block ID
     */
    void validate_block_id(uint32_t block_id) const;
};

}  // namespace npuw
}  // namespace ov
