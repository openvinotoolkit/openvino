// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "cpp/ie_cnn_network.h"
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

namespace ov {
namespace intel_gpu {

class CompiledModel : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<CompiledModel> Ptr;

    CompiledModel(InferenceEngine::CNNNetwork &network, InferenceEngine::RemoteContext::Ptr context, const ExecutionConfig& config,
                  InferenceEngine::InputsDataMap* inputs = nullptr, InferenceEngine::OutputsDataMap* outputs = nullptr);
    CompiledModel(cldnn::BinaryInputBuffer& ib, InferenceEngine::RemoteContext::Ptr context, const ExecutionConfig& config,
                  InferenceEngine::InputsDataMap* inputs = nullptr, InferenceEngine::OutputsDataMap* outputs = nullptr);

    void Export(std::ostream& networkModel) override;
    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                       const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    template <class T>
    InferenceEngine::IInferRequestInternal::Ptr GetInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                    const std::vector<std::shared_ptr<const ov::Node>>& outputs);
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;

    std::vector<std::shared_ptr<Graph>> m_graphs;
    InferenceEngine::RemoteContext::Ptr m_context;
    ExecutionConfig m_config;
    InferenceEngine::ITaskExecutor::Ptr m_taskExecutor;
    InferenceEngine::ITaskExecutor::Ptr m_waitExecutor;
    InferenceEngine::CNNNetwork m_network;
};

}  // namespace intel_gpu
}  // namespace ov
