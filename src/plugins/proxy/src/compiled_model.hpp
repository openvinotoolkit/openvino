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
    CompiledModel(const ie::SoExecutableNetworkInternal& model) : m_exec_network(model) {}

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override {
        return m_exec_network->CreateInferRequest();
    }

    void setProperty(const std::map<std::string, std::string>& properties) {
        IE_THROW(NotImplemented);
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
