// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "graph_context.h"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "sub_memory_manager.hpp"

namespace ov {
namespace intel_cpu {

class CompiledModel : public ov::ICompiledModel {
public:
    typedef std::shared_ptr<CompiledModel> Ptr;

    struct GraphGuard : public Graph {
        std::mutex _mutex;
        struct Lock : public std::unique_lock<std::mutex> {
            explicit Lock(GraphGuard& graph) : std::unique_lock<std::mutex>(graph._mutex), _graph(graph) {}
            GraphGuard& _graph;
        };
    };

public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  Config cfg,
                  const bool loaded_from_cache,
                  std::shared_ptr<SubMemoryManager> sub_memory_manager = nullptr);

    ~CompiledModel();

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    ov::Any get_property(const std::string& name) const override;

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("It's not possible to set property of an already compiled model. "
                                       "Set property to Core::compile_model during compilation");
    };

    void release_memory() override;

    std::string name() const {
        return m_name;
    }

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    friend class CompiledModelHolder;

    const std::shared_ptr<ov::Model> m_model;
    const std::shared_ptr<const ov::IPlugin> m_plugin;
    std::shared_ptr<ov::threading::ITaskExecutor> m_task_executor = nullptr;      //!< Holds a task executor
    std::shared_ptr<ov::threading::ITaskExecutor> m_callback_executor = nullptr;  //!< Holds a callback executor

    // Generic synchronization primitive on CompiledModel level.
    // Usage example: helps to avoid data races during CPU Graph initialization in multi-streams scenario
    std::shared_ptr<std::mutex> m_mutex;
    Config m_cfg;
    mutable std::atomic_int m_numRequests = {0};
    std::string m_name;

    const bool m_loaded_from_cache;
    // WARNING: Do not use m_graphs directly.
    mutable std::deque<GraphGuard> m_graphs;
    mutable SocketsWeights m_socketWeights;

    /* WARNING: Use get_graph() function to get access to graph in current stream.
     * NOTE: Main thread is interpreted as master thread of external stream so use this function to get access to graphs
     *       even from main thread
     */
    GraphGuard::Lock get_graph() const;

    std::vector<std::shared_ptr<CompiledModel>> get_sub_compiled_models() const {
        return m_sub_compiled_models;
    }

    std::vector<std::shared_ptr<CompiledModel>> m_sub_compiled_models;
    std::shared_ptr<SubMemoryManager> m_sub_memory_manager = nullptr;
    bool m_has_sub_compiled_models = false;
};

// This class provides safe access to the internal CompiledModel structures and helps to decouple SyncInferRequest and
// the CompiledModel internal structures
class CompiledModelHolder {
public:
    CompiledModelHolder(std::shared_ptr<const CompiledModel> compiled_model)
        : m_compiled_model(std::move(compiled_model)) {
        OPENVINO_ASSERT(!m_compiled_model->m_graphs.empty(),
                        "No graph was found in the compiled model: ",
                        m_compiled_model->name());
        m_graph = &(m_compiled_model->get_graph()._graph);
        m_id = (m_compiled_model->m_numRequests)++;
    }

    ~CompiledModelHolder() {
        if (m_compiled_model) {
            --(m_compiled_model->m_numRequests);
        }
    }

    CompiledModelHolder(const CompiledModelHolder&) = delete;
    CompiledModelHolder& operator=(const CompiledModelHolder&) = delete;

    CompiledModelHolder(CompiledModelHolder&&) = default;
    CompiledModelHolder& operator=(CompiledModelHolder&&) = default;

    const Graph& graph() const {
        return *m_graph;
    }

    CompiledModel::GraphGuard::Lock lock() {
        auto lock = m_compiled_model->get_graph();
        m_graph = &(lock._graph);
        OPENVINO_ASSERT(m_graph, "Graph ptr null check failed");
        return lock;
    }

    std::string name() const {
        return m_compiled_model->name();
    }

    std::shared_ptr<const ov::ICompiledModel> compiled_model() const {
        return m_compiled_model;
    }

    int id() const {
        return m_id;
    }

private:
    std::shared_ptr<const CompiledModel> m_compiled_model;
    const Graph* m_graph;
    int m_id;
};

}  // namespace intel_cpu
}  // namespace ov
