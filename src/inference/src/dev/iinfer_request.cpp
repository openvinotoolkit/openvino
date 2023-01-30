// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "iinfer_request.hpp"

#include "openvino/runtime/variable_state.hpp"

#define CHECK_INFER_REQUEST OPENVINO_ASSERT(m_async_request || m_sync_request, "Infer request is not initialized!")

ov::IInferRequest::IInferRequest(const std::shared_ptr<ov::IAsyncInferRequest>& request) : m_async_request(request) {}
ov::IInferRequest::IInferRequest(const std::shared_ptr<ov::ISyncInferRequest>& request) : m_sync_request(request) {}

void ov::IInferRequest::infer() {
    OPENVINO_ASSERT(m_sync_request, "Infer is not supported by asynchronous infer request.");
    m_sync_request->infer();
}

std::vector<ov::ProfilingInfo> ov::IInferRequest::get_profiling_info() const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->get_profiling_info();
    } else {
        return m_sync_request->get_profiling_info();
    }
}

ov::Tensor ov::IInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->get_tensor(port);
    } else {
        return m_sync_request->get_tensor(port);
    }
}

void ov::IInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        m_async_request->set_tensor(port, tensor);
    } else {
        m_sync_request->set_tensor(port, tensor);
    }
}

std::vector<ov::Tensor> ov::IInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->get_tensors(port);
    } else {
        return m_sync_request->get_tensors(port);
    }
}

void ov::IInferRequest::set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        m_async_request->set_tensors(port, tensors);
    } else {
        m_sync_request->set_tensors(port, tensors);
    }
}

std::vector<ov::VariableState> ov::IInferRequest::query_state() const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->query_state();
    } else {
        return m_sync_request->query_state();
    }
}

void ov::IInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        m_async_request->set_callback(callback);
    } else {
        m_sync_request->set_callback(callback);
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::IInferRequest::get_inputs() const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->m_sync_request->get_inputs();
    } else {
        return m_sync_request->get_inputs();
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::IInferRequest::get_outputs() const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->m_sync_request->get_outputs();
    } else {
        return m_sync_request->get_outputs();
    }
}

const std::shared_ptr<ov::ICompiledModel>& ov::IInferRequest::get_compiled_model() const {
    CHECK_INFER_REQUEST;
    if (m_async_request) {
        return m_async_request->m_sync_request->get_compiled_model();
    } else {
        return m_sync_request->get_compiled_model();
    }
}

void ov::IInferRequest::start_async() {
    OPENVINO_ASSERT(m_async_request, "Start async is not supported by synchronous infer request.");
    m_async_request->start_async();
}

void ov::IInferRequest::wait() {
    OPENVINO_ASSERT(m_async_request, "Wait is not supported by synchronous infer request.");
    m_async_request->wait();
}
bool ov::IInferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    OPENVINO_ASSERT(m_async_request, "Wait for timeout is not supported by synchronous infer request.");
    return m_async_request->wait_for(timeout);
}

void ov::IInferRequest::cancel() {
    OPENVINO_ASSERT(m_async_request, "Cancel is not supported by synchronous infer request.");
    m_async_request->cancel();
}
