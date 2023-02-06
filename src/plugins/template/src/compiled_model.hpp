// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "template_config.hpp"
#include "template_infer_request.hpp"

namespace TemplatePlugin {

class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                  const Configuration& cfg);

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    virtual ov::Any get_property(const std::string& name) const override;

    ov::RemoteContext get_context() const override;
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class TemplateInferRequest;
    friend class Plugin;

    void compile_model(const std::shared_ptr<ov::Model>& model);
    std::shared_ptr<const Plugin> get_template_plugin() const;

    std::atomic<std::size_t> _requestId = {0};
    Configuration _cfg;
    std::shared_ptr<ov::Model> m_model;
    std::map<std::string, std::size_t> _inputIndex;
    std::map<std::string, std::size_t> _outputIndex;
};
// ! [executable_network:header]

}  // namespace TemplatePlugin
