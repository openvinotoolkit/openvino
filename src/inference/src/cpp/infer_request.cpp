// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/infer_request.hpp"

#include <map>
#include <memory>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "transformations/utils/utils.hpp"

#define OV_INFER_REQ_CALL_STATEMENT(...)                                    \
    OPENVINO_ASSERT(_impl != nullptr, "InferRequest was not initialized."); \
    try {                                                                   \
        __VA_ARGS__;                                                        \
    } catch (const ov::Busy&) {                                             \
        throw;                                                              \
    } catch (const ov::Cancelled&) {                                        \
        throw;                                                              \
    } catch (const std::exception& ex) {                                    \
        OPENVINO_THROW(ex.what());                                          \
    } catch (...) {                                                         \
        OPENVINO_THROW("Unexpected exception");                             \
    }

namespace {

inline bool getPort(ov::Output<const ov::Node>& res_port,
                    const std::string& name,
                    const std::vector<std::vector<ov::Output<const ov::Node>>>& vector_ports) {
    for (const auto& ports : vector_ports) {
        for (const auto& port : ports) {
            const auto& names = port.get_names();
            if (names.find(name) != names.end()) {
                res_port = port;
                return true;
            }
        }
    }
    return false;
}

}  // namespace

namespace ov {

InferRequest::~InferRequest() {
    _impl = {};
}

InferRequest::InferRequest(const std::shared_ptr<ov::IAsyncInferRequest>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "InferRequest was not initialized.");
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({ _impl->set_tensor(port, get_tensor_impl(tensor)); });
}

void InferRequest::set_tensor(const ov::Output<ov::Node>& port, const Tensor& tensor) {
    set_tensor(ov::Output<const ov::Node>(port.get_node(), port.get_index()), tensor);
}

void InferRequest::set_tensor(const std::string& name, const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({
        ov::Output<const ov::Node> port;
        OPENVINO_ASSERT(::getPort(port, name, {_impl->get_inputs(), _impl->get_outputs()}),
                        "Port for tensor name " + name + " was not found.");
        set_tensor(port, tensor);
    });
}

void InferRequest::set_tensors(const std::string& name, const std::vector<Tensor>& tensors) {
    OV_INFER_REQ_CALL_STATEMENT({
        ov::Output<const ov::Node> port;
        OPENVINO_ASSERT(::getPort(port, name, {_impl->get_inputs()}),
                        "set_tensors error. Input port for tensor name ",
                        name,
                        " was not found.");
        set_tensors(port, tensors);
    })
}

void InferRequest::set_tensors(const ov::Output<const ov::Node>& port, const std::vector<Tensor>& tensors) {
    std::vector<ov::SoPtr<ov::ITensor>> tensor_ptrs;
    tensor_ptrs.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        tensor_ptrs.emplace_back(get_tensor_impl(tensor));
    }
    OV_INFER_REQ_CALL_STATEMENT({ _impl->set_tensors(port, tensor_ptrs); })
}

void InferRequest::set_input_tensor(size_t idx, const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto& inputs = _impl->get_inputs();
        OPENVINO_ASSERT(inputs.size() > idx,
                        "Input port for index ",
                        idx,
                        " was not found! The model has only ",
                        inputs.size(),
                        " inputs.");
        set_tensor(inputs.at(idx), tensor);
    });
}

void InferRequest::set_input_tensor(const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto& inputs = _impl->get_inputs();
        OPENVINO_ASSERT(inputs.size() == 1,
                        "set_input_tensor() must be called on a function with exactly one parameter.");
        set_tensor(inputs.at(0), tensor);
    });
}

void InferRequest::set_input_tensors(size_t idx, const std::vector<Tensor>& tensors) {
    OV_INFER_REQ_CALL_STATEMENT({
        OPENVINO_ASSERT(idx < _impl->get_inputs().size(),
                        "set_input_tensors error. Input port for index ",
                        idx,
                        " is out of bounds. Model has only ",
                        _impl->get_inputs().size(),
                        " inputs");
        set_tensors(_impl->get_inputs().at(idx), tensors);
    })
}

void InferRequest::set_input_tensors(const std::vector<Tensor>& tensors) {
    OV_INFER_REQ_CALL_STATEMENT({
        OPENVINO_ASSERT(_impl->get_inputs().size() == 1,
                        "set_input_tensors(tensors) must be used for single-input models only. Model has ",
                        _impl->get_inputs().size(),
                        " inputs");
        set_tensors(_impl->get_inputs().at(0), tensors);
    })
}

void InferRequest::set_output_tensor(size_t idx, const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto& outputs = _impl->get_outputs();
        OPENVINO_ASSERT(outputs.size() > idx,
                        "Output port for index ",
                        idx,
                        " was not found! The model has only ",
                        outputs.size(),
                        " outputs.");
        set_tensor(outputs.at(idx), tensor);
    });
}

void InferRequest::set_output_tensor(const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto& outputs = _impl->get_outputs();
        OPENVINO_ASSERT(outputs.size() == 1,
                        "set_output_tensor() must be called on a function with exactly one parameter.");
        set_tensor(outputs.at(0), tensor);
    });
}

Tensor InferRequest::get_tensor(const ov::Output<const ov::Node>& port) {
    OV_INFER_REQ_CALL_STATEMENT({
        OPENVINO_ASSERT(_impl->get_tensors(port).empty(),
                        "get_tensor shall not be used together with batched "
                        "set_tensors/set_input_tensors for port '",
                        port,
                        "'");
        auto tensor = _impl->get_tensor(port);
        if (!tensor._so)
            tensor._so = _so;

        return make_tensor(tensor);
    });
}

Tensor InferRequest::get_tensor(const ov::Output<ov::Node>& port) {
    return get_tensor(ov::Output<const ov::Node>(port.get_node(), port.get_index()));
}

Tensor InferRequest::get_tensor(const std::string& name) {
    OV_INFER_REQ_CALL_STATEMENT({
        ov::Output<const ov::Node> port;
        OPENVINO_ASSERT(::getPort(port, name, {_impl->get_inputs(), _impl->get_outputs()}),
                        "Port for tensor name " + name + " was not found.");
        return get_tensor(port);
    });
}

Tensor InferRequest::get_input_tensor(size_t idx) {
    OV_INFER_REQ_CALL_STATEMENT({ return get_tensor(_impl->get_inputs().at(idx)); });
}

Tensor InferRequest::get_output_tensor(size_t idx) {
    OV_INFER_REQ_CALL_STATEMENT({ return get_tensor(_impl->get_outputs().at(idx)); });
}

Tensor InferRequest::get_input_tensor() {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto& inputs = _impl->get_inputs();
        OPENVINO_ASSERT(inputs.size() == 1,
                        "get_input_tensor() must be called on a function with exactly one parameter.");
        return get_tensor(inputs.at(0));
    });
}

Tensor InferRequest::get_output_tensor() {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto& outputs = _impl->get_outputs();
        OPENVINO_ASSERT(outputs.size() == 1,
                        "get_output_tensor() must be called on a function with exactly one parameter.");
        return get_tensor(outputs.at(0));
    });
}

void InferRequest::infer() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->infer());
}

void InferRequest::cancel() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->cancel());
}

std::vector<ProfilingInfo> InferRequest::get_profiling_info() const {
    OV_INFER_REQ_CALL_STATEMENT(return _impl->get_profiling_info());
}

void InferRequest::start_async() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->start_async());
}

void InferRequest::wait() {
    OPENVINO_ASSERT(_impl != nullptr, "InferRequest was not initialized.");
    try {
        _impl->wait();
    } catch (const ov::Cancelled&) {
        throw;
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception");
    }
}

bool InferRequest::wait_for(const std::chrono::milliseconds timeout) {
    OPENVINO_ASSERT(_impl != nullptr, "InferRequest was not initialized.");
    try {
        return _impl->wait_for(timeout);
    } catch (const ov::Cancelled&) {
        throw;
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception");
    }
}

void InferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    OV_INFER_REQ_CALL_STATEMENT(_impl->set_callback(std::move(callback));)
}

std::vector<VariableState> InferRequest::query_state() {
    std::vector<VariableState> variable_states;
    OV_INFER_REQ_CALL_STATEMENT({
        for (auto&& state : _impl->query_state()) {
            if (!state._so)
                state._so = _so;
            variable_states.emplace_back(ov::VariableState{state._ptr, state._so});
        }
    })
    return variable_states;
}

void InferRequest::reset_state(){OV_INFER_REQ_CALL_STATEMENT({
    for (auto&& state : _impl->query_state()) {
        state->reset();
    }
})}

CompiledModel InferRequest::get_compiled_model() {
    OV_INFER_REQ_CALL_STATEMENT(return {std::const_pointer_cast<ICompiledModel>(_impl->get_compiled_model()), _so});
}

bool InferRequest::operator!() const noexcept {
    return !_impl;
}

InferRequest::operator bool() const noexcept {
    return (!!_impl);
}

bool InferRequest::operator!=(const InferRequest& r) const noexcept {
    return !(r == *this);
}

bool InferRequest::operator==(const InferRequest& r) const noexcept {
    return r._impl == _impl;
}

}  // namespace ov
