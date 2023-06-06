// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "remote_context.hpp"

namespace ov {
namespace proxy {

class InferRequest : public ov::IAsyncInferRequest {
private:
    std::shared_ptr<ov::IAsyncInferRequest> m_infer_request;
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;

public:
    InferRequest(std::shared_ptr<ov::IAsyncInferRequest>&& request,
                 const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_infer_request(std::move(request)),
          m_compiled_model(compiled_model) {}
    void start_async() override {
        m_infer_request->start_async();
    }

    void wait() override {
        m_infer_request->wait();
    }

    bool wait_for(const std::chrono::milliseconds& timeout) override {
        return m_infer_request->wait_for(timeout);
    }

    void cancel() override {
        m_infer_request->cancel();
    }

    void set_callback(std::function<void(std::exception_ptr)> callback) override {
        m_infer_request->set_callback(callback);
    }

    void infer() override {
        m_infer_request->infer();
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_infer_request->get_profiling_info();
    }

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override {
        auto tensor = m_infer_request->get_tensor(port);
        if (tensor.is<ov::RemoteTensor>()) {
            auto remote_context = std::dynamic_pointer_cast<ov::proxy::RemoteContext>(m_compiled_model->get_context());
            OPENVINO_ASSERT(remote_context);
            tensor = remote_context->wrap_tensor(tensor.as<ov::RemoteTensor>());
        }
        return tensor;
    }

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override {
        m_infer_request->set_tensor(port, tensor);
    }

    std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const override {
        auto tensors = m_infer_request->get_tensors(port);
        for (auto&& tensor : tensors) {
            if (tensor.is<ov::RemoteTensor>()) {
                auto remote_context =
                    std::dynamic_pointer_cast<ov::proxy::RemoteContext>(m_compiled_model->get_context());
                OPENVINO_ASSERT(remote_context);
                tensor = remote_context->wrap_tensor(tensor.as<ov::RemoteTensor>());
            }
        }
        return tensors;
    }

    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) override {
        return m_infer_request->set_tensors(port, tensors);
    }

    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override {
        return m_infer_request->query_state();
    }

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override {
        return m_compiled_model;
    }

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return m_infer_request->get_inputs();
    }

    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return m_infer_request->get_outputs();
    }
};

}  // namespace proxy
}  // namespace ov
