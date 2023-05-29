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

// #include "executable.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace hetero {

// forward declaration
class CompiledModel;

class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model);
    ~InferRequest();

    void infer() override;
    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    void start_pipeline();
    void wait_pipeline();
    void cancel();

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override;

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override;

    void check_tensors() const  override {
        // NOTE: ignore `check_tensor` of inputs and outputs of Hetero Compiled Model because 
        // `m_tensors` are not allocated 
        return;
    }

private:
    friend class CompiledModel;
    friend class AsyncInferRequest;

    std::shared_ptr<const CompiledModel> get_hetero_model() const;

    
    enum { StartPipeline, WaitPipeline, numOfStages };
    
    struct SubRequestDesc {
        ov::SoPtr<ov::ICompiledModel> _network;
        ov::SoPtr<ov::IAsyncInferRequest> _request;
        std::array<openvino::itt::handle_t, numOfStages> _profilingTask;
    };

    std::vector<SubRequestDesc> m_infer_requests;

    // vurusovs: NOTE to connect subrequests
    std::map<ov::Output<const ov::Node>, ov::Tensor> m_port_to_tensor_map;
    std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::IAsyncInferRequest>> m_port_to_request_map;

    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages> m_durations;
};

}  // namespace hetero
}  // namespace ov
