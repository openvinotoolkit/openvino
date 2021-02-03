// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

namespace MKLDNNPlugin {

class MKLDNNExecNetwork;

class MKLDNNInferRequest : public InferenceEngine::InferRequestInternal {
public:
    typedef std::shared_ptr<MKLDNNInferRequest> Ptr;
    explicit MKLDNNInferRequest(InferenceEngine::InputsDataMap      networkInputs,
                                InferenceEngine::OutputsDataMap     networkOutputs,
                                std::shared_ptr<MKLDNNExecNetwork>  execNetwork);

    ~MKLDNNInferRequest() override;

    void InferImpl() override;

    InferenceEngine::StatusCode Cancel() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;

    void SetBatch(int batch = -1) override;

    std::vector<InferenceEngine::IVariableStateInternal::Ptr> QueryState() override;

private:
    void PushInputData();
    void PushStates();
    void PullStates();

    void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision dataType);

    void changeDefaultPtr();
    std::shared_ptr<MKLDNNExecNetwork>  execNetwork;
    MKLDNNGraph*                        graph = nullptr;
    std::map<std::string, void*>        externalPtr;
    openvino::itt::handle_t             profilingTask;
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> memoryStates;
};
}  // namespace MKLDNNPlugin
