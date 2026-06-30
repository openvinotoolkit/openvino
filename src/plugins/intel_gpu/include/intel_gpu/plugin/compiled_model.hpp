// Copyright (C) 2018-2026 Intel Corporation
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

    // Builds the simplified v1 GPU compatibility descriptor.
    // Format: meta=<ver>;ov=<ov>;desc=[<driver/hw features>]
    static std::string build_runtime_requirements(const cldnn::device_info& info);

    // Layout version of the runtime requirements descriptor persisted in the blob.
    // Bump when the on-disk contract (the data following the magic marker) changes so the
    // importer can detect and reject descriptors produced by a future build.
    static constexpr uint32_t runtime_requirements_version = 1;

    // Magic marker that prefixes the compatibility-descriptor block in the exported blob, letting
    // the importer reject blobs that lack it (e.g. produced by an OpenVINO build predating this
    // feature) instead of misreading their input count. Chosen far larger than any realistic input
    // count so such a blob's first post-cache_mode word can never collide with it ("OVEP_RRQ" in ASCII).
    static constexpr uint64_t runtime_requirements_magic = 0x4F5645505F525251ULL;

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

    // Helper function to return the model name for ITT tracing
    std::string_view get_model_name() const {
        return m_model_name;
    }

    const std::vector<std::shared_ptr<Graph>>& get_graphs() const;
    std::shared_ptr<Graph> get_graph(size_t n) const;

    void release_memory() override;
    void set_backing_tensor(const std::shared_ptr<ov::Tensor>& tensor) {
        _backing_tensor = tensor;
    }

private:
    RemoteContextImpl::Ptr m_context;
    ExecutionConfig m_config;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
    std::string m_model_name;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    std::vector<std::shared_ptr<Graph>> m_graphs;
    bool m_loaded_from_cache;
    std::string m_runtime_requirements;
    std::shared_ptr<ov::Tensor> _backing_tensor;
};

}  // namespace ov::intel_gpu
