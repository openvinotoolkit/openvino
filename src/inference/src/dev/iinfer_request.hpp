// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {

class IInferRequest {
public:
    IInferRequest(const std::shared_ptr<ov::IAsyncInferRequest>& request);
    IInferRequest(const std::shared_ptr<ov::ISyncInferRequest>& request);
    void infer();

    std::vector<ov::ProfilingInfo> get_profiling_info() const;

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor);

    std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const;
    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors);

    std::vector<ov::VariableState> query_state() const;

    void set_callback(std::function<void(std::exception_ptr)> callback);

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const;
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const;

    const std::shared_ptr<ov::ICompiledModel>& get_compiled_model() const;

    void start_async();

    void wait();
    bool wait_for(const std::chrono::milliseconds& timeout);

    void cancel();

private:
    std::shared_ptr<ov::IAsyncInferRequest> m_async_request;
    std::shared_ptr<ov::ISyncInferRequest> m_sync_request;
};

}  // namespace ov
