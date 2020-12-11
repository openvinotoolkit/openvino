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
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <threading/ie_itask_executor.hpp>
#include <openvino/itt.hpp>

#include <ngraph/runtime/tensor.hpp>
#include <executable.hpp>

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

private:
    void allocateDeviceBuffers();
    void allocateBlobs();

    enum {
        Preprocess,
        Postprocess,
        StartPipeline,
        WaitPipeline,
        numOfStages
    };

    std::shared_ptr<ExecutableNetwork>                      _executableNetwork;
    std::array<openvino::itt::handle_t, numOfStages>        _profilingTask;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages>   _durations;

    InferenceEngine::BlobMap                                _networkOutputBlobs;
    ngraph::ParameterVector                                 _parameters;
    ngraph::ResultVector                                    _results;

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>>   _inputTensors;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>>   _outputTensors;
    std::shared_ptr<ngraph::runtime::Executable>            _executable;
};
// ! [infer_request:header]

}  // namespace TemplatePlugin
