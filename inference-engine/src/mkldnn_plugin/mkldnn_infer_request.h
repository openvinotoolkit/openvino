// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

namespace MKLDNNPlugin {

class MKLDNNInferRequest : public InferenceEngine::InferRequestInternal {
public:
    typedef std::shared_ptr<MKLDNNInferRequest> Ptr;
    explicit MKLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs,
                          InferenceEngine::OutputsDataMap networkOutputs);

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

    void SetGraph(const MKLDNNGraph::Ptr& graph);

    void SetBatch(int batch = -1) override;

private:
    template <typename T> void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob);

    void changeDefaultPtr();
    MKLDNNGraph::Ptr graph;
    std::map<std::string, void*> externalPtr;

    int m_curBatch;
};
}  // namespace MKLDNNPlugin
