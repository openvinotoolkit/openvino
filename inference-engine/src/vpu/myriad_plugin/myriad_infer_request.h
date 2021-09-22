// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>

#include <ie_common.h>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>

#include <vpu/utils/logger.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/graph_transformer.hpp>

#include "myriad_executor.h"

namespace vpu {
namespace MyriadPlugin {

class MyriadInferRequest : public InferenceEngine::IInferRequestInternal {
    MyriadExecutorPtr _executor;
    Logger::Ptr _log;
    std::vector<StageMetaInfo> _stagesMetaData;
    PluginConfiguration _config;

    const DataInfo _inputInfo;
    const DataInfo _outputInfo;

    GraphDesc _graphDesc;
    std::vector<uint8_t> resultBuffer;
    std::vector<uint8_t> inputBuffer;
    std::map<std::string, ie::Blob::Ptr> _constDatas;
    bool _isNetworkConstant;

public:
    typedef std::shared_ptr<MyriadInferRequest> Ptr;

    explicit MyriadInferRequest(GraphDesc &_graphDesc,
                                InferenceEngine::InputsDataMap networkInputs,
                                InferenceEngine::OutputsDataMap networkOutputs,
                                DataInfo& compilerInputsInfo,
                                DataInfo& compilerOutputsInfo,
                                const std::vector<StageMetaInfo> &blobMetaData,
                                const PluginConfiguration &myriadConfig,
                                const Logger::Ptr &log,
                                const MyriadExecutorPtr &executor,
                                std::map<std::string, ie::Blob::Ptr> constDatas,
                                bool isNetworkConstant);

    void InferImpl() override;
    void InferAsync();
    void GetResult();

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>
    GetPerformanceCounts() const override;
};

}  // namespace MyriadPlugin
}  // namespace vpu
