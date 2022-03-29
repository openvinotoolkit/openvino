// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "ie_icore.hpp"
#include "mock_infer_request.hpp"
#include "openvino/runtime/compiled_model.hpp"

class MockCompiledModel : public InferenceEngine::IExecutableNetworkInternal {
public:
    MockCompiledModel(const std::shared_ptr<const ov::Model>& model, const std::map<std::string, std::string>& config)
        : m_model(model),
          m_config(config) {
        std::vector<std::shared_ptr<const ov::Node>> inputs, outputs;
        for (const auto& input : m_model->inputs())
            inputs.emplace_back(input.get_node_shared_ptr());
        setInputs(inputs);
        for (const auto& output : m_model->outputs())
            outputs.emplace_back(output.get_node_shared_ptr());
        setOutputs(outputs);
    }

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<MockInferRequest>(networkInputs,
                                                  networkOutputs,
                                                  std::static_pointer_cast<MockCompiledModel>(shared_from_this()));
    }
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override {
        return std::make_shared<MockInferRequest>(inputs,
                                                  outputs,
                                                  std::static_pointer_cast<MockCompiledModel>(shared_from_this()));
    }

    void setProperty(const std::map<std::string, std::string>& properties) {
        IE_THROW(NotImplemented);
    }

    InferenceEngine::Parameter GetConfig(const std::string& name) const override {
        IE_THROW(NotImplemented);
    }

    InferenceEngine::Parameter GetMetric(const std::string& name) const override {
        IE_THROW(NotImplemented);
    }

    std::shared_ptr<ov::Model> GetExecGraphInfo() override {
        IE_THROW(NotImplemented);
    }

    void Export(std::ostream& modelStream) override {
        IE_THROW(NotImplemented);
    }

private:
    friend MockInferRequest;
    const std::shared_ptr<const ov::Model> m_model;
    std::map<std::string, std::string> m_config;
};
