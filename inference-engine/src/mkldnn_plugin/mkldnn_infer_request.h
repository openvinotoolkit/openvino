// Copyright (C) 2018-2020 Intel Corporation
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

    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;

    /**
     * @brief Given optional implementation of setting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) override;

    /**
     * @brief Given optional implementation of getting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) override;

    void SetBatch(int batch = -1) override;

    std::vector<InferenceEngine::IVariableStateInternal::Ptr> QueryState() override;

private:
    void PushInputData();

    void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision dataType);

    void changeDefaultPtr();
    std::shared_ptr<MKLDNNExecNetwork>  execNetwork;
    MKLDNNGraph*                        graph = nullptr;
    std::map<std::string, void*>        externalPtr;
    openvino::itt::handle_t             profilingTask;
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> memoryStates;
};
}  // namespace MKLDNNPlugin
