// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/iplugin_managed_cache.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace tp {

/// \brief The paged-attention KV cache of a tensor-parallel model, managed by the plugin.
///
/// One KEY_VALUE group in the model-level representation: key_cache.N / value_cache.N with the model's KV head
/// count. Each rank holds the slice of its KV heads in its own device memory; block indices mean the same blocks
/// on every rank, so every operation is the same request repeated per rank and layer. The cache is bound to every
/// infer request of the compiled model as soon as it exists, and unbound when it is freed.
class CacheController : public ov::IPluginManagedCache {
public:
    /// Null when the rank models have no cache ports (no paged attention).
    /// \param kv_replicas Consecutive ranks holding the same KV heads (fewer KV heads than ranks).
    static std::shared_ptr<CacheController> create(const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled,
                                                   const std::vector<std::string>& device_names,
                                                   std::shared_ptr<const ov::ICore> core,
                                                   size_t kv_replicas);

    std::vector<ov::CacheGroupInfo> get_groups() const override;
    size_t get_available_memory() const override;
    void allocate(size_t group, size_t num_blocks, bool zero_initialize) override;
    void reallocate(size_t group, size_t num_blocks, bool zero_initialize_new_blocks) override;
    void deallocate(size_t group) override;
    size_t get_num_allocated_blocks(size_t group) const override;
    void copy_blocks(size_t group, const std::map<size_t, std::list<size_t>>& block_copy_map) override;
    void zero_blocks(size_t group, const std::set<size_t>& block_indices) override;

    /// Binds the cache, if any, to a new infer request's rank requests and keeps doing so until unregistered.
    uint64_t register_request(const std::vector<ov::SoPtr<ov::IAsyncInferRequest>>& rank_requests);
    void unregister_request(uint64_t id);

    /// Throws unless the cache has been allocated.
    void check_allocated() const;

    /// The cache ports of rank 0, in the order of the group's tensors.
    const std::vector<ov::Output<const ov::Node>>& ports() const {
        return m_ranks.front().ports;
    }

private:
    using RankTensors = std::vector<std::vector<ov::SoPtr<ov::ITensor>>>;  // [rank][tensor]

    struct Rank {
        std::string device;
        ov::SoPtr<ov::IRemoteContext> context;
        std::vector<ov::Output<const ov::Node>> ports;
        std::vector<ov::element::Type> precisions;
        std::vector<ov::Shape> shapes;  ///< 0 blocks
        size_t block_bytes = 0;         ///< one block of all tensors of the rank
        size_t max_tensor_block_bytes = 0;
    };

    CacheController() = default;
    void check_group(size_t group) const;
    /// The complete cache with `num_blocks` blocks, blocks [0, keep) copied from the current one; touches no member.
    RankTensors build(size_t num_blocks, size_t keep, bool zero_new) const;
    /// Binds `tensors` (or zero-block placeholders when empty) to every registered request.
    void bind_all(const RankTensors& tensors);
    void bind(const std::vector<std::weak_ptr<ov::IAsyncInferRequest>>& request, const RankTensors& tensors) const;
    RankTensors placeholders() const;
    void zero(const ov::SoPtr<ov::ITensor>& tensor, size_t first_block, size_t num_blocks) const;

    std::vector<Rank> m_ranks;
    std::shared_ptr<const ov::ICore> m_core;
    ov::CacheGroupInfo m_group;

    mutable std::mutex m_mutex;
    RankTensors m_tensors;
    size_t m_num_blocks = 0;
    std::map<uint64_t, std::vector<std::weak_ptr<ov::IAsyncInferRequest>>> m_requests;
    uint64_t m_next_request = 0;
};

using CacheControllerPtr = std::shared_ptr<CacheController>;

}  // namespace tp
}  // namespace ov
