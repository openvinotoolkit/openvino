// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tp_gpu/paged_attention_cache_controller.hpp"

namespace ov {
namespace tp_gpu {

/// \brief The physical KV cache of a tensor-parallel paged-attention model.
///
/// Each rank owns the slice of the cache that belongs to its KV heads, in its
/// own device memory. Block indices are logical and mean the same thing on
/// every rank, so growing, duplicating and clearing blocks is the same request
/// repeated per rank.
///
/// The controller only owns the tensors; binding them to the per-rank infer
/// requests is the infer request's job, which is why it can be created before
/// any request exists. `generation()` tells a request whether the tensors it
/// last bound are still the current ones.
class CacheController : public IPagedAttentionCacheController {
public:
    /// Builds a controller over the cache ports of the given rank models, or
    /// returns null when they have none -- which is the normal case: only
    /// models converted to paged attention carry a cache the plugin can own.
    static std::shared_ptr<CacheController> create(
        const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled);

    void allocate_cache_if_needed(size_t num_blocks) override;
    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override;
    void zero_blocks(const std::set<size_t>& block_indices) override;
    void clear() override;

    size_t get_num_layers() const override { return m_num_layers; }
    size_t get_num_cache_tensors() const override;
    size_t get_block_size() const override { return m_block_size; }
    size_t get_block_size_in_bytes() const override { return m_block_size_in_bytes; }
    size_t get_num_allocated_blocks() const override { return m_allocated_blocks; }

    /// The cache ports of one rank, in a fixed order.
    const std::vector<ov::Output<const ov::Node>>& ports(size_t rank) const {
        return m_ranks.at(rank).ports;
    }

    /// The tensors currently bound to those ports. Empty until the first
    /// allocation, and replaced wholesale whenever the cache grows.
    const std::vector<ov::SoPtr<ov::ITensor>>& tensors(size_t rank) const {
        return m_ranks.at(rank).tensors;
    }

    size_t world_size() const { return m_ranks.size(); }

    /// Bumped on every reallocation. A request that bound tensors at an older
    /// generation is holding tensors that no longer belong to the cache.
    uint64_t generation() const { return m_generation; }

private:
    struct Rank {
        ov::SoPtr<ov::IRemoteContext> context;
        std::vector<ov::Output<const ov::Node>> ports;
        std::vector<ov::element::Type> precisions;
        /// Port shapes with the block count still open at dimension 0.
        std::vector<ov::Shape> shapes;
        std::vector<ov::SoPtr<ov::ITensor>> tensors;
    };

    std::vector<Rank> m_ranks;
    size_t m_num_layers = 0;
    size_t m_block_size = 0;
    size_t m_block_size_in_bytes = 0;
    size_t m_allocated_blocks = 0;
    uint64_t m_generation = 0;
};

using CacheControllerPtr = std::shared_ptr<CacheController>;

}  // namespace tp_gpu
}  // namespace ov
