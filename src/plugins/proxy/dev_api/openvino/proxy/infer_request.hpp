// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace proxy {

class InferRequest : public ov::IAsyncInferRequest {
private:
    ov::SoPtr<ov::IAsyncInferRequest> m_infer_request;
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;

public:
    InferRequest(ov::SoPtr<ov::IAsyncInferRequest>&& request,
                 const std::shared_ptr<const ov::ICompiledModel>& compiled_model);
    void start_async() override;

    void wait() override;

    bool wait_for(const std::chrono::milliseconds& timeout) override;

    void cancel() override;

    void set_callback(std::function<void(std::exception_ptr)> callback) override;

    void infer() override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;

    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override;

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override;

    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override;

    const ov::SoPtr<ov::IAsyncInferRequest> get_hardware_request() const;
};

}  // namespace proxy
}  // namespace ov
