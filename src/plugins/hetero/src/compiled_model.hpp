// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config.hpp"
#include "hsm_reader_handler.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin.hpp"
#include "remote_context.hpp"
#include "subgraph_collector.hpp"

namespace ov {
namespace hetero {

class Plugin;
class InferRequest;

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::vector<ov::hetero::SubmodelInfo>& compiled_submodels,
                  const SubgraphsMappingInfo& mapping_info,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  ov::hetero::RemoteContext::Ptr context,
                  const Configuration& cfg);

    CompiledModel(std::istream& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const Configuration& cfg,
                  const bool loaded_from_cache);

    /// Imports an HSM blob; the tensor must hold the whole container.
    CompiledModel(const ov::Tensor& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const Configuration& cfg,
                  const bool loaded_from_cache);

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    const std::vector<ov::Output<const ov::Node>>& inputs() const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

private:
    friend class InferRequest;

    void compile_model(const std::vector<ov::hetero::SubmodelInfo>& submodels);

    std::shared_ptr<const Plugin> get_hetero_plugin() const;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void set_inputs_and_outputs();

    void export_container(std::ostream& model_stream) const;
    void import_container(const ov::runtime::hsm::Reader& reader, const HsmBlobSource& source);

    ov::AnyMap get_load_config(const std::string& device) const;

    Configuration m_cfg;
    std::string m_name;
    const bool m_loaded_from_cache;
    std::vector<ov::Output<const ov::Node>> m_compiled_inputs;
    std::vector<ov::Output<const ov::Node>> m_compiled_outputs;
    SubgraphsMappingInfo m_mapping_info;

    struct CompiledModelDesc {
        std::string device;
        std::shared_ptr<ov::Model> model;
        ov::SoPtr<ov::ICompiledModel> compiled_model;
    };
    std::vector<CompiledModelDesc> m_compiled_submodels;
};
}  // namespace hetero
}  // namespace ov
