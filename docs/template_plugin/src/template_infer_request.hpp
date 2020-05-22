// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include <ie_common.h>
#include <ie_profiling.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <threading/ie_itask_executor.hpp>

#include "template_config.hpp"

namespace TemplatePlugin {

class ExecutableNetwork;

// ! [infer_request:header]
class TemplateInferRequest : public InferenceEngine::InferRequestInternal {
public:
    typedef std::shared_ptr<TemplateInferRequest> Ptr;

    TemplateInferRequest(const InferenceEngine::InputsDataMap&     networkInputs,
                         const InferenceEngine::OutputsDataMap&    networkOutputs,
                         const std::shared_ptr<ExecutableNetwork>& executableNetwork);
    ~TemplateInferRequest() override;

    void InferImpl() override;
    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void inferPreprocess();
    void startPipeline();
    void waitPipeline();
    void inferPostprocess();

    std::shared_ptr<ExecutableNetwork>                      _executableNetwork;

private:
    void allocateDeviceBuffers();
    void allocateInputBlobs();
    void allocateOutputBlobs();

    enum {
        Preprocess,
        Postprocess,
        StartPipeline,
        WaitPipeline,
        numOfStages
    };

    std::array<InferenceEngine::ProfilingTask, numOfStages> _profilingTask;

    InferenceEngine::BlobMap                                _inputsNCHW;
    InferenceEngine::BlobMap                                _outputsNCHW;

    // for performance counts
    double                                                  _inputPreprocessTime   = 0.0;
    double                                                  _inputTransferTime     = 0.0;
    double                                                  _executeTime           = 0.0;
    double                                                  _outputTransferTime    = 0.0;
    double                                                  _outputPostProcessTime = 0.0;
};
// ! [infer_request:header]

}  // namespace TemplatePlugin
