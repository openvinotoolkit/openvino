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
    typedef std::shared_ptr<MKLDNNInferRequest> Ptr;
    explicit MKLDNNInferRequest(InferenceEngine::InputsDataMap      networkInputs,
                                InferenceEngine::OutputsDataMap     networkOutputs,
                                std::shared_ptr<MKLDNNExecNetwork>  execNetwork);

    MKLDNNInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                       const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                       std::shared_ptr<MKLDNNExecNetwork>                  execNetwork);

    ~MKLDNNInferRequest();

    void InferImpl() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;

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

private:
    void CreateInferRequest();
    void PushInputData();
    void PushStates();
    void PullStates();
    void redefineMemoryForInputNodes();

    void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision dataType);

    void changeDefaultPtr();
    std::shared_ptr<MKLDNNExecNetwork>  execNetwork;
    MKLDNNGraph*                        graph = nullptr;
    std::map<std::string, void*>        externalPtr;
    openvino::itt::handle_t             profilingTask;
    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> memoryStates;
    MKLDNNAsyncInferRequest*            _asyncRequest = nullptr;
};
}  // namespace MKLDNNPlugin
