// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <ngraph/function.hpp>

#include "template_async_infer_request.hpp"
#include "template_infer_request.hpp"
#include "template_rw_properties.hpp"

namespace TemplatePlugin {

// forward declaration
class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    ExecutableNetwork(const std::shared_ptr<const ngraph::Function>& function,
                      const InferenceEngine::InputsDataMap& inputInfoMap,
                      const InferenceEngine::OutputsDataMap& outputsInfoMap,
                      const std::map<std::string, std::string>& cfg,
                      const std::shared_ptr<Plugin>& plugin);

    ExecutableNetwork(std::istream& model,
                      const std::map<std::string, std::string>& cfg,
                      const std::shared_ptr<Plugin>& plugin);

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    void Export(std::ostream& model) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

private:
    friend class TemplateInferRequest;
    friend class Plugin;

    void CompileNetwork(const std::shared_ptr<const ngraph::Function>& function,
                        const InferenceEngine::InputsDataMap& inputInfoMap,
                        const InferenceEngine::OutputsDataMap& outputsInfoMap);
    void InitExecutor();

    void init_properties(const std::map<std::string, std::string>& cfg);

    std::atomic<std::size_t> _requestId = {0};
    RwProperties _cfg;
    std::shared_ptr<Plugin> _plugin;
    std::shared_ptr<ngraph::Function> _function;
    std::map<std::string, std::size_t> _inputIndex;
    std::map<std::string, std::size_t> _outputIndex;
};
// ! [executable_network:header]

}  // namespace TemplatePlugin
