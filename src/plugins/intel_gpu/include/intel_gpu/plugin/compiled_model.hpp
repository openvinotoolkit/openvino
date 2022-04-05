// Copyright (C) 2018-2022 Intel Corporation
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
#include "intel_gpu/plugin/device_config.hpp"
#include "intel_gpu/plugin/remote_context.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

class CompiledModel : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<CompiledModel> Ptr;

    CompiledModel(InferenceEngine::CNNNetwork &network, std::shared_ptr<InferenceEngine::RemoteContext> context, Config config);

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                       const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;

    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;

    std::vector<std::shared_ptr<Graph>> m_graphs;
    InferenceEngine::gpu::ClContext::Ptr m_context;
    Config m_config;
    InferenceEngine::ITaskExecutor::Ptr m_taskExecutor;
    InferenceEngine::ITaskExecutor::Ptr m_waitExecutor;
};

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
