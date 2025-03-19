// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/proxy/infer_request.hpp"

#include <memory>
#include <openvino/runtime/iremote_tensor.hpp>

#include "openvino/proxy/plugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "remote_context.hpp"

ov::proxy::InferRequest::InferRequest(ov::SoPtr<ov::IAsyncInferRequest>&& request,
                                      const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
      m_infer_request(std::move(request)),
      m_compiled_model(compiled_model) {}
void ov::proxy::InferRequest::start_async() {
    m_infer_request->start_async();
}

void ov::proxy::InferRequest::wait() {
    m_infer_request->wait();
}

bool ov::proxy::InferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    return m_infer_request->wait_for(timeout);
}

void ov::proxy::InferRequest::cancel() {
    m_infer_request->cancel();
}

void ov::proxy::InferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    m_infer_request->set_callback(callback);
}

void ov::proxy::InferRequest::infer() {
    m_infer_request->infer();
}

std::vector<ov::ProfilingInfo> ov::proxy::InferRequest::get_profiling_info() const {
    return m_infer_request->get_profiling_info();
}

ov::SoPtr<ov::ITensor> ov::proxy::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto tensor = m_infer_request->get_tensor(port);
    if (!tensor._so)
        tensor._so = m_infer_request._so;
    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
        auto remote_context = std::dynamic_pointer_cast<ov::proxy::RemoteContext>(m_compiled_model->get_context()._ptr);
        OPENVINO_ASSERT(remote_context);
        tensor = remote_context->wrap_tensor(tensor);
    }
    return tensor;
}

void ov::proxy::InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    m_infer_request->set_tensor(port, ov::proxy::get_hardware_tensor(tensor, true));
}

std::vector<ov::SoPtr<ov::ITensor>> ov::proxy::InferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    auto tensors = m_infer_request->get_tensors(port);
    for (auto&& tensor : tensors) {
        if (!tensor._so)
            tensor._so = m_infer_request._so;
        if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
            auto remote_context =
                std::dynamic_pointer_cast<ov::proxy::RemoteContext>(m_compiled_model->get_context()._ptr);
            OPENVINO_ASSERT(remote_context);
            tensor = remote_context->wrap_tensor(tensor);
        }
    }
    return tensors;
}

void ov::proxy::InferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    std::vector<ov::SoPtr<ov::ITensor>> hw_tensors;
    for (auto& tensor : tensors) {
        hw_tensors.push_back(ov::proxy::get_hardware_tensor(tensor, true));
    }
    return m_infer_request->set_tensors(port, hw_tensors);
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::proxy::InferRequest::query_state() const {
    auto states = m_infer_request->query_state();
    for (auto&& state : states) {
        if (!state._so)
            state._so = m_infer_request._so;
    }
    return states;
}

const std::shared_ptr<const ov::ICompiledModel>& ov::proxy::InferRequest::get_compiled_model() const {
    return m_compiled_model;
}

const std::vector<ov::Output<const ov::Node>>& ov::proxy::InferRequest::get_inputs() const {
    return m_infer_request->get_inputs();
}

const std::vector<ov::Output<const ov::Node>>& ov::proxy::InferRequest::get_outputs() const {
    return m_infer_request->get_outputs();
}

const ov::SoPtr<ov::IAsyncInferRequest> ov::proxy::InferRequest::get_hardware_request() const {
    return m_infer_request;
}
