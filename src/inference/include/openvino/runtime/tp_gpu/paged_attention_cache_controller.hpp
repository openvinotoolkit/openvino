// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contract between the TP_GPU plugin and a paged-attention pipeline
 *        over the physical KV cache.
 * @file openvino/runtime/tp_gpu/paged_attention_cache_controller.hpp
 */

#pragma once

#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <set>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp_gpu {

/**
 * @brief Owns the physical KV cache of a tensor-parallel compiled model.
 *
 * A paged-attention pipeline normally allocates the cache itself, through the
 * remote context of the compiled model. That does not work under tensor
 * parallelism: the cache is split by KV head across the rank devices, so it is
 * neither one allocation nor one device, and only the plugin knows the split.
 * The pipeline therefore hands cache ownership over and drives it through this
 * interface instead.
 *
 * Sizes reported here describe the cache as a whole -- summed over the ranks --
 * because that is what a scheduler budgets against. Block indices are logical
 * and identical on every rank: the ranks hold different heads of the same
 * blocks, never different blocks.
 *
 * A controller belongs to one compiled model and is shared by its infer
 * requests, which is what a paged-attention pipeline wants: one cache, one
 * scheduler, one request driving it.
 *
 * @note Not thread-safe. The pipeline is expected to drive the cache and the
 *       inference from the same thread, which is what makes the reported
 *       allocation state meaningful.
 */
class IPagedAttentionCacheController {
public:
    virtual ~IPagedAttentionCacheController() = default;

    /**
     * @brief Grow the cache to hold at least @p num_blocks blocks.
     *
     * A no-op when it already does. Growing preserves the blocks already in
     * the cache.
     */
    virtual void allocate_cache_if_needed(size_t num_blocks) = 0;

    /**
     * @brief Duplicate blocks inside the cache.
     *
     * Used for copy-on-write when a forked sequence writes to a block it was
     * sharing.
     *
     * @param block_copy_map Source block index -> destination block indices.
     */
    virtual void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) = 0;

    /**
     * @brief Zero the given blocks before a new sequence reuses them.
     */
    virtual void zero_blocks(const std::set<size_t>& block_indices) = 0;

    /**
     * @brief Release the cache entirely.
     */
    virtual void clear() = 0;

    /// @return Number of decoder layers the cache covers.
    virtual size_t get_num_layers() const = 0;

    /// @return Number of cache tensors, counting every rank and both key and
    ///         value tensors of every layer.
    virtual size_t get_num_cache_tensors() const = 0;

    /// @return Tokens per block.
    virtual size_t get_block_size() const = 0;

    /// @return Bytes one block occupies across all layers and all ranks.
    virtual size_t get_block_size_in_bytes() const = 0;

    /// @return Blocks currently allocated.
    virtual size_t get_num_allocated_blocks() const = 0;
};

using PagedAttentionCacheControllerPtr = std::shared_ptr<IPagedAttentionCacheController>;

/**
 * @brief Read-only property of a TP_GPU compiled model carrying its cache
 *        controller.
 *
 * Present only on models the plugin recognized as paged-attention. A pipeline
 * that finds it must allocate the cache through the controller instead of
 * through the remote context; one that does not find it is talking to a plugin
 * that manages no cache of its own, and keeps allocating as before.
 */
static constexpr Property<PagedAttentionCacheControllerPtr, PropertyMutability::RO>
    paged_attention_cache_controller{"TP_GPU_PAGED_ATTENTION_CACHE_CONTROLLER"};

}  // namespace tp_gpu
}  // namespace ov
