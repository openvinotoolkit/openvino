// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "ie_icore.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace proxy {
class CompiledModel : public InferenceEngine::IExecutableNetworkInternal {
public:
    CompiledModel(const ie::SoExecutableNetworkInternal& model) : m_exec_network(model) {
        _parameters = m_exec_network->getInputs();
        _results = m_exec_network->getOutputs();
        for (const auto& it : m_exec_network->GetInputsInfo()) {
            _networkInputs[it.first] = std::const_pointer_cast<InferenceEngine::InputInfo>(it.second);
        }
        for (const auto& it : m_exec_network->GetOutputsInfo()) {
            _networkOutputs[it.first] = std::const_pointer_cast<InferenceEngine::Data>(it.second);
        }
    }

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override {
        return m_exec_network->CreateInferRequest();
    }

    InferenceEngine::Parameter GetConfig(const std::string& name) const override {
        return m_exec_network->GetConfig(name);
    }

    InferenceEngine::Parameter GetMetric(const std::string& name) const override {
        return m_exec_network->GetMetric(name);
    }

    std::shared_ptr<ov::Model> GetExecGraphInfo() override {
        return m_exec_network->GetExecGraphInfo();
    }

    void Export(std::ostream& modelStream) override {
        return m_exec_network->Export(modelStream);
    }

private:
    ie::SoExecutableNetworkInternal m_exec_network;
};

}  // namespace proxy
}  // namespace ov
