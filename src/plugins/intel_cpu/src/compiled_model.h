// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "extension_mngr.h"
#include "graph.h"
#include "graph_context.h"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/threading/thread_local.hpp"

namespace ov {
namespace intel_cpu {

class CompiledModel : public ov::ICompiledModel {
public:
    typedef std::shared_ptr<CompiledModel> Ptr;

    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::Model>& orig_model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const Config& cfg,
                  const ExtensionManager::Ptr& extMgr,
                  const bool loaded_from_cache = false);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    ov::Any get_property(const std::string& name) const override;

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "CompiledModel::set_property is not supported by this plugin!");
    };

    const std::shared_ptr<const ov::Model>& get_orig_model() const {
        return _original_model;
    }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

protected:
    friend class SyncInferRequest;

    const std::shared_ptr<ov::Model> _model;
    const std::shared_ptr<const ov::Model> _original_model;
    std::vector<std::shared_ptr<ov::IVariableState>> _memory_states;
    const std::shared_ptr<const ov::IPlugin> _plugin;
    std::shared_ptr<ov::threading::ITaskExecutor> _taskExecutor = nullptr;      //!< Holds a task executor
    std::shared_ptr<ov::threading::ITaskExecutor> _callbackExecutor = nullptr;  //!< Holds a callback executor

    // Generic synchronization primitive on CompiledModel level.
    // Usage example: helps to avoid data races during CPU Graph initialization in multi-streams scenario
    mutable std::shared_ptr<std::mutex> _mutex;
    Config _cfg;
    ExtensionManager::Ptr extensionManager;
    mutable std::atomic_int _numRequests = {0};
    std::string _name;
    struct GraphGuard : public Graph {
        std::mutex _mutex;
        struct Lock : public std::unique_lock<std::mutex> {
            explicit Lock(GraphGuard& graph) : std::unique_lock<std::mutex>(graph._mutex), _graph(graph) {}
            GraphGuard& _graph;
        };
    };

    const bool _loaded_from_cache;
    // WARNING: Do not use _graphs directly.
    mutable std::deque<GraphGuard> _graphs;
    mutable NumaNodesWeights _numaNodesWeights;

    /* WARNING: Use GetGraph() function to get access to graph in current stream.
     * NOTE: Main thread is interpreted as master thread of external stream so use this function to get access to graphs
     *       even from main thread
     */
    GraphGuard::Lock GetGraph() const;

    bool CanProcessDynBatch(const std::shared_ptr<ov::Model> &model) const;

    ov::Any GetConfigLegacy(const std::string& name) const;
    ov::Any GetMetric(const std::string& name) const;
    ov::Any GetMetricLegacy(const std::string& name, const GraphGuard& graph) const;
};

}   // namespace intel_cpu
}   // namespace ov

