// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/icompiled_model.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>

namespace ov {
namespace intel_gpu {

class CompiledModel : public ov::ICompiledModel {
public:
    using Ptr = std::shared_ptr<CompiledModel>;

    CompiledModel(std::shared_ptr<ov::Model> model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  RemoteContextImpl::Ptr context,
                  const ExecutionConfig& config);
    CompiledModel(cldnn::BinaryInputBuffer ib,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  RemoteContextImpl::Ptr context,
                  const ExecutionConfig& config);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    ov::Any get_property(const std::string& name) const override;

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("Not Implemented: CompiledModel::set_property is not supported by this plugin!");
    };

    const std::vector<ov::Output<const ov::Node>>& outputs() const override { return m_outputs; }
    const std::vector<ov::Output<const ov::Node>>& inputs() const override { return m_inputs;}

    bool is_new_api() const { return std::static_pointer_cast<const ov::intel_gpu::Plugin>(get_plugin())->is_new_api(); }
    RemoteContextImpl::Ptr get_context_impl() const {return m_context; }
    const std::vector<std::shared_ptr<Graph>>& get_graphs() const;
    std::shared_ptr<Graph> get_graph(size_t n) const;

private:
    RemoteContextImpl::Ptr m_context;
    ExecutionConfig m_config;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
    std::shared_ptr<ov::Model> m_model;
    std::string m_model_name;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    std::vector<std::shared_ptr<Graph>> m_graphs;
    bool m_loaded_from_cache;
};

}  // namespace intel_gpu
}  // namespace ov
