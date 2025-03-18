// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace ov::intel_gpu {

class CompiledModel : public ov::ICompiledModel {
public:
    using Ptr = std::shared_ptr<CompiledModel>;

    CompiledModel(std::shared_ptr<ov::Model> model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  RemoteContextImpl::Ptr context,
                  const ExecutionConfig& config);
    CompiledModel(cldnn::BinaryInputBuffer& ib,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  RemoteContextImpl::Ptr context,
                  const ExecutionConfig& config,
                  const bool loaded_from_cache);
    ~CompiledModel() {
        auto streams_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(get_task_executor());
        streams_executor->cpu_reset();
    }

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    ov::Any get_property(const std::string& name) const override;

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("It's not possible to set property of an already compiled model. Set property "
                                       "to Core::compile_model during compilation");
    };

    const std::vector<ov::Output<const ov::Node>>& outputs() const override {
        return m_outputs;
    }
    const std::vector<ov::Output<const ov::Node>>& inputs() const override {
        return m_inputs;
    }

    RemoteContextImpl::Ptr get_context_impl() const {
        return m_context;
    }
    const std::vector<std::shared_ptr<Graph>>& get_graphs() const;
    std::shared_ptr<Graph> get_graph(size_t n) const;

    void release_memory() override;

private:
    RemoteContextImpl::Ptr m_context;
    ExecutionConfig m_config;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
    std::string m_model_name;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    std::vector<std::shared_ptr<Graph>> m_graphs;
    bool m_loaded_from_cache;
};

}  // namespace ov::intel_gpu
