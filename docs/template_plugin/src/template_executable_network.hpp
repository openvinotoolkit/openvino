// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <ngraph/function.hpp>

#include "template_config.hpp"
#include "template_infer_request.hpp"
#include "template_async_infer_request.hpp"

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

namespace TemplatePlugin {

class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    ExecutableNetwork(const std::shared_ptr<const ngraph::Function>& function,
                      const Configuration&                           cfg,
                      const std::shared_ptr<Plugin>&                 plugin);

    ExecutableNetwork(std::istream&                  model,
                      const Configuration&           cfg,
                      const std::shared_ptr<Plugin>& plugin);

    ~ExecutableNetwork() override = default;

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    void ExportImpl(std::ostream& model) override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;
    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;
    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

private:
    friend class TemplateInferRequest;

    void CompileNetwork(const std::shared_ptr<const ngraph::Function>& function);
    void InitExecutor();

    std::atomic<std::size_t>                    _requestId = {0};
    Configuration                               _cfg;
    std::shared_ptr<Plugin>                     _plugin;
    std::shared_ptr<ngraph::Function>           _function;
    std::map<std::string, std::size_t>          _inputIndex;
    std::map<std::string, std::size_t>          _outputIndex;
};
// ! [executable_network:header]

}  // namespace TemplatePlugin
