// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <map>
#include <memory>
#include <openvino/itt.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace HeteroPlugin {

class HeteroInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    typedef std::shared_ptr<HeteroInferRequest> Ptr;

    struct SubRequestDesc {
        InferenceEngine::SoExecutableNetworkInternal _network;
        InferenceEngine::SoIInferRequestInternal _request;
        openvino::itt::handle_t _profilingTask;
    };
    using SubRequestsList = std::vector<SubRequestDesc>;

    HeteroInferRequest(InferenceEngine::InputsDataMap networkInputs,
                       InferenceEngine::OutputsDataMap networkOutputs,
                       const SubRequestsList& inferRequests,
                       const std::unordered_map<std::string, std::string>& blobNameMap);

    HeteroInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& networkInputs,
                       const std::vector<std::shared_ptr<const ov::Node>>& networkOutputs,
                       const SubRequestsList& inferRequests,
                       const std::unordered_map<std::string, std::string>& blobNameMap);

    void InferImpl() override;

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& blob) override;

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;

    void SetBlob(const std::string& name,
                 const InferenceEngine::Blob::Ptr& blob,
                 const InferenceEngine::PreProcessInfo& info) override;

    const InferenceEngine::PreProcessInfo& GetPreProcess(const std::string& name) const override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    SubRequestsList _inferRequests;
    std::map<std::string, InferenceEngine::Blob::Ptr> _blobs;
    std::map<std::string, InferenceEngine::IInferRequestInternal*> _subRequestFromBlobName;

private:
    void CreateInferRequest(const std::unordered_map<std::string, std::string>& subgraphInputToOutputBlobNames);
};

}  // namespace HeteroPlugin
