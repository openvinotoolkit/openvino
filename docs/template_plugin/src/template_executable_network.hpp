// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <utility>
#include <tuple>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cnn_network_impl.hpp>
#include <threading/ie_itask_executor.hpp>

#include <ngraph/function.hpp>

#include "template_config.hpp"
#include "template_infer_request.hpp"
#include "template_async_infer_request.hpp"

namespace TemplatePlugin {

class Engine;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    ExecutableNetwork(InferenceEngine::ICNNNetwork&  network,
                      const Configuration&           cfg);

    ExecutableNetwork(std::istream &                 model,
                      const Configuration&           cfg);

    ~ExecutableNetwork() override = default;

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    void ExportImpl(std::ostream& model) override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;
    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;
    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    std::atomic<std::size_t>                    _requestId = {0};
    std::string                                 _name;
    Configuration                               _cfg;

private:
    void CompileGraph(const std::shared_ptr<const ngraph::Function> & ngraphFunction);

    std::shared_ptr<Engine>                     _plugin;
    InferenceEngine::ITaskExecutor::Ptr         _waitExecutor;
};
// ! [executable_network:header]

}  // namespace TemplatePlugin
