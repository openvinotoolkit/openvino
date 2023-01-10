// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "executable.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/iinfer_request.hpp"

namespace TemplatePlugin {

// forward declaration
class ExecutableNetwork;

// ! [infer_request:header]
class TemplateInferRequest : public ov::IInferRequest {
public:
    typedef std::shared_ptr<TemplateInferRequest> Ptr;

    TemplateInferRequest(const std::shared_ptr<ExecutableNetwork>& executableNetwork);
    ~TemplateInferRequest();

    void infer() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override;
    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void infer_preprocess();
    void start_pipeline();
    void wait_pipeline();
    void infer_postprocess();

private:
    std::shared_ptr<ExecutableNetwork> get_template_model() const;

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::array<openvino::itt::handle_t, numOfStages> _profilingTask;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages> _durations;

    InferenceEngine::BlobMap _networkOutputBlobs;

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> _inputTensors;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> _outputTensors;
    std::shared_ptr<ngraph::runtime::Executable> _executable;
};
// ! [infer_request:header]

}  // namespace TemplatePlugin
