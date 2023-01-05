// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime AsyncInferRequest interface
 * @file iasync_nfer_request.hpp
 */

#pragma once

#include <memory>

#include "openvino/runtime/iinfer_request.hpp"
#include "threading/ie_itask_executor.hpp"

namespace ov {

class IAsyncInferRequest : public IInferRequest {
public:
    IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                       const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                       const InferenceEngine::ITaskExecutor::Ptr& callback_executor);

    void infer() override;
    void start_async() override;

    void wait() override;
    bool wait_for(const std::chrono::milliseconds timeout) override;

    void cancel() override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    ov::Tensor get_input_tensor(size_t idx) const override;
    void set_input_tensor(size_t idx, const ov::Tensor& tensor) override;

    std::vector<ov::Tensor> get_input_tensors(size_t idx) const override;
    void set_input_tensors(size_t idx, const std::vector<ov::Tensor>& tensors) override;
    void set_input_tensors_imp(size_t idx, const std::vector<ov::Tensor>& tensors) override;

    ov::Tensor get_output_tensor(size_t idx) const override;
    void set_output_tensor(size_t idx, const ov::Tensor& tensor) override;

    std::vector<ov::VariableState> query_state() const override;

    void set_callback(std::function<void(std::exception_ptr)> callback) override;

    void check_tensors() override;
};

}  // namespace ov
