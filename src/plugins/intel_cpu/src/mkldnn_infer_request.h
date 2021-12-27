// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace MKLDNNPlugin {

class MKLDNNExecNetwork;
class MKLDNNAsyncInferRequest;

class MKLDNNInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    ~MKLDNNInferRequest();

    void InferImpl() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void SetBatch(int batch = -1) override;

    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override;

    /**
     * @brief      Sets the pointer to asynchronous inference request that holds this request
     * @param[in]  asyncRequest Pointer to asynchronous inference request
     */
    void SetAsyncRequest(MKLDNNAsyncInferRequest* asyncRequest);

    /**
     * @brief If `_asyncRequest` is initialized throw exception with `InferenceEngine::INFER_CANCELLED` status if inference request is canceled
     */
    void ThrowIfCanceled() const;

protected:
    explicit MKLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                InferenceEngine::OutputsDataMap networkOutputs,
                                std::shared_ptr<MKLDNNExecNetwork> execNetwork_)
    : IInferRequestInternal(networkInputs, networkOutputs), execNetwork(execNetwork_) {}

    explicit MKLDNNInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                std::shared_ptr<MKLDNNExecNetwork> execNetwork_)
    : IInferRequestInternal(inputs, outputs), execNetwork(execNetwork_) {}

    void CreateInferRequest();
    void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision dataType);

    virtual void fillBlobs() = 0;
    virtual void PushInputData() = 0;

    MKLDNNGraph* graph = nullptr;
    std::map<std::string, void*> externalPtr;

private:
    void PushStates();
    void PullStates();
    void redefineMemoryForInputNodes();

    void changeDefaultPtr();
    std::shared_ptr<MKLDNNExecNetwork>  execNetwork;
    openvino::itt::handle_t             profilingTask;
    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> memoryStates;
    MKLDNNAsyncInferRequest*            _asyncRequest = nullptr;
};

class MKLDNNInferRequestOldApi : public MKLDNNInferRequest {
public:
    explicit MKLDNNInferRequestOldApi(InferenceEngine::InputsDataMap networkInputs,
                                      InferenceEngine::OutputsDataMap networkOutputs,
                                      std::shared_ptr<MKLDNNExecNetwork> execNetwork);

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
private:
    void PushInputData() override;
    void fillBlobs() override;
};

class MKLDNNInferRequestNewApi : public MKLDNNInferRequest {
public:
    explicit MKLDNNInferRequestNewApi(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                      const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                      std::shared_ptr<MKLDNNExecNetwork> execNetwork);

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
private:
    void PushInputData() override;
    void fillBlobs() override;

    std::unordered_map<std::string, std::shared_ptr<const ov::Node>> modelInputsMap;
    std::unordered_map<std::string, std::shared_ptr<const ov::Node>> modelOutputsMap;
};

}  // namespace MKLDNNPlugin
