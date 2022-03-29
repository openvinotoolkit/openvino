// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "ie_icore.hpp"
#include "openvino/runtime/compiled_model.hpp"

class MockCompiledModel;

class MockInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    MockInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                     const InferenceEngine::OutputsDataMap& networkOutputs,
                     const std::shared_ptr<MockCompiledModel>& executableNetwork)
        : InferenceEngine::IInferRequestInternal(networkInputs, networkOutputs),
          m_compiled_model(executableNetwork) {
        allocate_blobs();
    }
    MockInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                     const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                     const std::shared_ptr<MockCompiledModel>& executableNetwork)
        : InferenceEngine::IInferRequestInternal(inputs, outputs),
          m_compiled_model(executableNetwork) {
        allocate_blobs();
    }
    ~MockInferRequest() = default;

    void InferImpl() override;
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) override;

private:
    std::shared_ptr<MockCompiledModel> m_compiled_model;
    std::map<std::string, InferenceEngine::Blob::Ptr> m_inputs;
    std::map<std::string, InferenceEngine::Blob::Ptr> m_outputs;
    void allocate_blobs();
};

