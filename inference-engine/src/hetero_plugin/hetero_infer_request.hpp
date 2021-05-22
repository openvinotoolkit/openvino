// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <ie_common.h>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <openvino/itt.hpp>

namespace HeteroPlugin {

class HeteroInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    typedef std::shared_ptr<HeteroInferRequest> Ptr;

    struct SubRequestDesc {
        InferenceEngine::SoExecutableNetworkInternal  _network;
        InferenceEngine::SoIInferRequestInternal      _request;
        openvino::itt::handle_t                       _profilingTask;
    };
    using SubRequestsList = std::vector<SubRequestDesc>;

    explicit HeteroInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                InferenceEngine::OutputsDataMap networkOutputs,
                                const SubRequestsList &inferRequests,
                                const std::unordered_map<std::string, std::string>& blobNameMap);

    void InferImpl() override;

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void updateInOutIfNeeded();

    SubRequestsList _inferRequests;
    std::map<std::string, InferenceEngine::Blob::Ptr>   _blobs;
};

}  // namespace HeteroPlugin

