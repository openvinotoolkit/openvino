// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cache_eviction.hpp"
#include "cache_manager.hpp"
#include "config.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace template_plugin {

class Plugin;
class InferRequest;

/**
 * @class CompiledModel
 * @brief Implementation of compiled model
 */
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::SoPtr<ov::IRemoteContext>& context,
                  const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                  const Configuration& cfg,
                  bool loaded_from_cache = false);

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class InferRequest;
    friend class Plugin;

    void compile_model(const std::shared_ptr<ov::Model>& model);
    std::shared_ptr<const Plugin> get_template_plugin() const;

    // --- NEW: shared CacheManager (engine-scoped) ---
    mutable std::shared_ptr<ov::cache::CacheManager> m_cache_manager;
    mutable std::mutex m_cache_mgr_mutex;

    // Eviction defaults (exposed to requests)
    ov::cache::CacheEvictionConfig m_eviction_cfg{/*start*/ 32,
                                                  /*recent*/ 128,
                                                  /*max*/ 672,
                                                  ov::cache::AggregationMode::NORM_SUM,
                                                  /*apply_rotation*/ false,
                                                  /*snapkv_window*/ 8};

    // Create-or-get CacheManager (constructed from compiled model, no request coupling)
    std::shared_ptr<ov::cache::CacheManager> get_or_create_cache_manager_locked() const;

    mutable std::atomic<std::size_t> m_request_id = {0};
    Configuration m_cfg;
    std::shared_ptr<ov::Model> m_model;
    const bool m_loaded_from_cache;
};

}  // namespace template_plugin
}  // namespace ov
