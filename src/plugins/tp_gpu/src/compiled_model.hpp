// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/runtime/icompiled_model.hpp"
#include "cache_controller.hpp"
#include "rank_workers.hpp"
#include "tp_gpu/tp_device_coordinator.hpp"
#include "tp_l0_shared_context.hpp"

namespace ov {
namespace tp_gpu {

class CompiledModel : public ov::ICompiledModel {
public:
    /// \param model  The user model.  Null when the compiled model was
    ///        restored from a blob: there is no IR at that point, and
    ///        `inputs()`/`outputs()` fall back to rank 0 instead.
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  std::vector<ov::SoPtr<ov::ICompiledModel>>&& rank_compiled,
                  std::vector<std::string>&& device_names,
                  TPL0SharedContextPtr shared_l0_ctx = nullptr,
                  TPDeviceCoordinatorPtr device_coordinator = nullptr,
                  std::vector<std::string>&& sharded_state_ids = {},
                  bool loaded_from_cache = false);

    const std::vector<ov::Output<const ov::Node>>& inputs() const override;

    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    void release_memory() override;

    const std::vector<ov::SoPtr<ov::ICompiledModel>>& get_rank_compiled() const { return m_rank_compiled; }
    const std::vector<std::string>& get_device_names() const { return m_device_names; }

    /// Ids of the variables whose per-rank states each hold a slice of the KV
    /// cache along the kv-head axis.  A caller reading or writing whole state
    /// tensors has to see them gathered and scattered; every other variable is
    /// replicated and can be fanned out as is.
    const std::vector<std::string>& get_sharded_state_ids() const { return m_sharded_state_ids; }

    /// The paged-attention cache this model owns, or null when the model does
    /// not use one.  Shared by every infer request of this model: one cache,
    /// one scheduler driving it.
    const CacheControllerPtr& get_cache_controller() const { return m_cache_controller; }

    /// The rank rendezvous, Level Zero command lists and shared scratch arena
    /// support one outer inference at a time.  Rank execution inside that
    /// inference remains parallel.
    std::unique_lock<std::mutex> lock_inference() const {
        return std::unique_lock<std::mutex>(m_inference_mutex);
    }

    /// Threads the ranks run on.  Created on first use, which is always under
    /// lock_inference(), and kept for the life of the model: creating them per
    /// inference costs tens of microseconds of start-up skew, and that skew is
    /// paid again at every one of the model's AllReduce points.
    RankWorkers& rank_workers() const {
        if (!m_rank_workers) {
            m_rank_workers = std::make_unique<RankWorkers>(m_rank_compiled.size());
        }
        return *m_rank_workers;
    }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    // Lifetime: shared L0 context must outlive every rank's CompiledModel
    // (and the ze_engine/USM allocations they own). Declared first so it is
    // destroyed last. The device coordinator depends on the shared context
    // and is destroyed before it.
    TPL0SharedContextPtr m_shared_l0_ctx;
    TPDeviceCoordinatorPtr m_device_coordinator;
    std::vector<ov::SoPtr<ov::ICompiledModel>> m_rank_compiled;
    std::vector<std::string> m_device_names;
    std::vector<std::string> m_sharded_state_ids;
    CacheControllerPtr m_cache_controller;
    bool m_loaded_from_cache{false};
    mutable std::mutex m_inference_mutex;
    // Declared last so it is destroyed first: the worker threads must be
    // joined before anything they might still be touching goes away.
    mutable std::unique_ptr<RankWorkers> m_rank_workers;
};

}  // namespace tp_gpu
}  // namespace ov
