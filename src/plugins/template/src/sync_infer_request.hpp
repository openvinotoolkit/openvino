// Copyright (C) 2018-2023 Intel Corporation
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
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace template_plugin {

// forward declaration
class CompiledModel;

// ! [infer_request:header]
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& compiled_model);
    ~InferRequest();

    void infer() override;
    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void infer_preprocess();
    void start_pipeline();
    void wait_pipeline();
    void infer_postprocess();

    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) override;

private:
    std::shared_ptr<const CompiledModel> get_template_model() const;

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::array<openvino::itt::handle_t, numOfStages> m_profiling_task;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages> m_durations;

    std::vector<ov::Tensor> m_backend_input_tensors;
    std::vector<ov::Tensor> m_backend_output_tensors;
    std::shared_ptr<ov::runtime::Executable> m_executable;
};
// ! [infer_request:header]

}  // namespace template_plugin
}  // namespace ov
