// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/proxy/infer_request.hpp"

#include <memory>
#include <openvino/runtime/iremote_tensor.hpp>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "remote_context.hpp"

ov::proxy::InferRequest::InferRequest(ov::SoPtr<ov::IAsyncInferRequest>&& request,
                                      const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
      m_infer_request(std::move(request)),
      m_compiled_model(compiled_model) {
    std::cout << "Proxy InferRquest created" << std::endl;
}
void ov::proxy::InferRequest::start_async() {
    std::cout << "Proxy start_async" << std::endl;
    m_infer_request->start_async();
}

void ov::proxy::InferRequest::wait() {
    std::cout << "Proxy wait" << std::endl;
    m_infer_request->wait();
}

bool ov::proxy::InferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    std::cout << "Proxy wait_for" << std::endl;
    return m_infer_request->wait_for(timeout);
}

void ov::proxy::InferRequest::cancel() {
    std::cout << "Proxy cancel" << std::endl;
    m_infer_request->cancel();
}

void ov::proxy::InferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    std::cout << "Proxy set_callback" << std::endl;
    m_infer_request->set_callback(callback);
}

void ov::proxy::InferRequest::infer() {
    std::cout << "Proxy infer" << std::endl;
    m_infer_request->infer();
}

std::vector<ov::ProfilingInfo> ov::proxy::InferRequest::get_profiling_info() const {
    std::cout << "Proxy profiling_info" << std::endl;
    return m_infer_request->get_profiling_info();
}

ov::SoPtr<ov::ITensor> ov::proxy::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::cout << "Proxy get_tensor" << std::endl;
    auto tensor = m_infer_request->get_tensor(port);
    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
        auto remote_context = std::dynamic_pointer_cast<ov::proxy::RemoteContext>(m_compiled_model->get_context()._ptr);
        OPENVINO_ASSERT(remote_context);
        tensor = remote_context->wrap_tensor(tensor);
    }
    return tensor;
}

void ov::proxy::InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    std::cout << "Proxy set_tensor" << std::endl;
    m_infer_request->set_tensor(port, tensor);
}

std::vector<ov::SoPtr<ov::ITensor>> ov::proxy::InferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    std::cout << "Proxy get_tensors" << std::endl;
    auto tensors = m_infer_request->get_tensors(port);
    for (auto&& tensor : tensors) {
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
    std::cout << "Proxy set_tensors" << std::endl;
    return m_infer_request->set_tensors(port, tensors);
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::proxy::InferRequest::query_state() const {
    std::cout << "Proxy query_state" << std::endl;
    auto states = m_infer_request->query_state();
    for (auto&& state : states) {
        if (!state._so)
            state._so = m_infer_request._so;
    }
    return states;
}

const std::shared_ptr<const ov::ICompiledModel>& ov::proxy::InferRequest::get_compiled_model() const {
    std::cout << "Proxy get_comp_model" << std::endl;
    return m_compiled_model;
}

const std::vector<ov::Output<const ov::Node>>& ov::proxy::InferRequest::get_inputs() const {
    std::cout << "Proxy infer inputs" << std::endl;
    return m_infer_request->get_inputs();
}

const std::vector<ov::Output<const ov::Node>>& ov::proxy::InferRequest::get_outputs() const {
    std::cout << "Proxy infer outputs" << std::endl;
    return m_infer_request->get_outputs();
}

const ov::SoPtr<ov::IAsyncInferRequest> ov::proxy::InferRequest::get_hardware_request() const {
    std::cout << "Proxy hw request" << std::endl;
    return m_infer_request;
}
