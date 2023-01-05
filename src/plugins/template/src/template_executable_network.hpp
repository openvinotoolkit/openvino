// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <ngraph/function.hpp>

#include "openvino/runtime/icompiled_model.hpp"
#include "template_async_infer_request.hpp"
#include "template_config.hpp"
#include "template_infer_request.hpp"

namespace TemplatePlugin {

// forward declaration
class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class ExecutableNetwork : public ov::ICompiledModel {
public:
    ExecutableNetwork(const std::shared_ptr<ov::Model>& model,
                      const Configuration& cfg,
                      const std::shared_ptr<Plugin>& plugin);

    ExecutableNetwork(std::istream& model, const Configuration& cfg, const std::shared_ptr<Plugin>& plugin);

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    void export_model(std::ostream& model) const override;
    InferenceEngine::IInferRequestInternal::Ptr create_infer_request_impl(
        const std::vector<ov::Output<const ov::Node>>& inputs,
        const std::vector<ov::Output<const ov::Node>>& outputs) const override;
    InferenceEngine::IInferRequestInternal::Ptr create_infer_request() const override;

    ov::Any get_property(const std::string& name) const override;

private:
    friend class TemplateInferRequest;
    friend class Plugin;

    void CompileNetwork(const std::shared_ptr<const ov::Model>& model,
                        const InferenceEngine::InputsDataMap& inputInfoMap,
                        const InferenceEngine::OutputsDataMap& outputsInfoMap);
    void InitExecutor();

    std::atomic<std::size_t> _requestId = {0};
    Configuration _cfg;
    std::shared_ptr<Plugin> _plugin;
    std::shared_ptr<ov::Model> m_model;
    std::map<std::string, std::size_t> _inputIndex;
    std::map<std::string, std::size_t> _outputIndex;
};
// ! [executable_network:header]

}  // namespace TemplatePlugin
