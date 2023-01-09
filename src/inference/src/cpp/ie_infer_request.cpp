// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_infer_request.hpp"

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "ie_infer_async_request_base.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_context.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "transformations/utils/utils.hpp"

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
namespace InferenceEngine {

#define INFER_REQ_CALL_STATEMENT(...)                                     \
    if (_impl == nullptr)                                                 \
        IE_THROW(NotAllocated) << "Inference Request is not initialized"; \
    try {                                                                 \
        __VA_ARGS__                                                       \
    } catch (...) {                                                       \
        ::InferenceEngine::details::Rethrow();                            \
    }

#define OV_INFER_REQ_CALL_STATEMENT(...)                                    \
    OPENVINO_ASSERT(_impl != nullptr, "InferRequest was not initialized."); \
    try {                                                                   \
        __VA_ARGS__;                                                        \
    } catch (const ::InferenceEngine::RequestBusy& ex) {                    \
        throw ov::Busy(ex.what());                                          \
    } catch (const std::exception& ex) {                                    \
        throw ov::Exception(ex.what());                                     \
    } catch (...) {                                                         \
        OPENVINO_ASSERT(false, "Unexpected exception");                     \
    }

InferRequest::~InferRequest() {
    _impl = {};
}

InferRequest::InferRequest(const IInferRequestInternal::Ptr& impl, const std::shared_ptr<void>& so)
    : _impl(impl),
      _so(so) {
    IE_ASSERT(_impl != nullptr);
}

IE_SUPPRESS_DEPRECATED_START

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    INFER_REQ_CALL_STATEMENT(_impl->SetBlob(name, data);)
}

Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    Blob::Ptr blobPtr;
    INFER_REQ_CALL_STATEMENT(blobPtr = _impl->GetBlob(name);)
    std::string error = "Internal error: blob with name `" + name + "` is not allocated!";
    if (blobPtr == nullptr)
        IE_THROW() << error;
    const bool remoteBlobPassed = blobPtr->is<RemoteBlob>();
    if (!remoteBlobPassed && blobPtr->buffer() == nullptr)
        IE_THROW() << error;
    return blobPtr;
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data, const PreProcessInfo& info) {
    INFER_REQ_CALL_STATEMENT(_impl->SetBlob(name, data, info);)
}

const PreProcessInfo& InferRequest::GetPreProcess(const std::string& name) const {
    INFER_REQ_CALL_STATEMENT(return _impl->GetPreProcess(name);)
}

void InferRequest::Infer() {
    INFER_REQ_CALL_STATEMENT(_impl->Infer();)
}

void InferRequest::Cancel() {
    INFER_REQ_CALL_STATEMENT(_impl->Cancel();)
}

std::map<std::string, InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    INFER_REQ_CALL_STATEMENT(return _impl->GetPerformanceCounts();)
}

void InferRequest::SetInput(const BlobMap& inputs) {
    INFER_REQ_CALL_STATEMENT(for (auto&& input : inputs) { _impl->SetBlob(input.first, input.second); })
}

void InferRequest::SetOutput(const BlobMap& results) {
    INFER_REQ_CALL_STATEMENT(for (auto&& result : results) { _impl->SetBlob(result.first, result.second); })
}

void InferRequest::SetBatch(const int batch) {
    INFER_REQ_CALL_STATEMENT(_impl->SetBatch(batch);)
}

void InferRequest::StartAsync() {
    INFER_REQ_CALL_STATEMENT(_impl->StartAsync();)
}

StatusCode InferRequest::Wait(int64_t millis_timeout) {
    INFER_REQ_CALL_STATEMENT(return _impl->Wait(millis_timeout);)
}

void InferRequest::SetCompletionCallbackImpl(std::function<void()> callbackToSet) {
    INFER_REQ_CALL_STATEMENT(_impl->SetCallback([callbackToSet](std::exception_ptr) {
        callbackToSet();
    });)
}

#define CATCH_IE_EXCEPTION_RETURN(StatusCode, ExceptionType) \
    catch (const ::InferenceEngine::ExceptionType&) {        \
        return StatusCode;                                   \
    }

#define CATCH_IE_EXCEPTIONS_RETURN                                   \
    CATCH_IE_EXCEPTION_RETURN(GENERAL_ERROR, GeneralError)           \
    CATCH_IE_EXCEPTION_RETURN(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_IE_EXCEPTION_RETURN(NETWORK_NOT_LOADED, NetworkNotLoaded)  \
    CATCH_IE_EXCEPTION_RETURN(PARAMETER_MISMATCH, ParameterMismatch) \
    CATCH_IE_EXCEPTION_RETURN(NOT_FOUND, NotFound)                   \
    CATCH_IE_EXCEPTION_RETURN(OUT_OF_BOUNDS, OutOfBounds)            \
    CATCH_IE_EXCEPTION_RETURN(UNEXPECTED, Unexpected)                \
    CATCH_IE_EXCEPTION_RETURN(REQUEST_BUSY, RequestBusy)             \
    CATCH_IE_EXCEPTION_RETURN(RESULT_NOT_READY, ResultNotReady)      \
    CATCH_IE_EXCEPTION_RETURN(NOT_ALLOCATED, NotAllocated)           \
    CATCH_IE_EXCEPTION_RETURN(INFER_NOT_STARTED, InferNotStarted)    \
    CATCH_IE_EXCEPTION_RETURN(NETWORK_NOT_READ, NetworkNotRead)      \
    CATCH_IE_EXCEPTION_RETURN(INFER_CANCELLED, InferCancelled)

void InferRequest::SetCompletionCallbackImpl(std::function<void(InferRequest, StatusCode)> callbackToSet) {
    INFER_REQ_CALL_STATEMENT(
        auto weakThis =
            InferRequest{std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*) {}}, _so};
        _impl->SetCallback([callbackToSet, weakThis](std::exception_ptr exceptionPtr) {
            StatusCode statusCode = StatusCode::OK;
            if (exceptionPtr != nullptr) {
                statusCode = [&] {
                    try {
                        std::rethrow_exception(exceptionPtr);
                    }
                    CATCH_IE_EXCEPTIONS_RETURN catch (const std::exception&) {
                        return GENERAL_ERROR;
                    }
                    catch (...) {
                        return UNEXPECTED;
                    }
                }();
            }
            callbackToSet(weakThis, statusCode);
        });)
}

void InferRequest::SetCompletionCallbackImpl(IInferRequest::CompletionCallback callbackToSet) {
    INFER_REQ_CALL_STATEMENT(
        IInferRequest::Ptr weakThis =
            InferRequest{std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*) {}}, _so};
        _impl->SetCallback([callbackToSet, weakThis](std::exception_ptr exceptionPtr) {
            StatusCode statusCode = StatusCode::OK;
            if (exceptionPtr != nullptr) {
                statusCode = [&] {
                    try {
                        std::rethrow_exception(exceptionPtr);
                    }
                    CATCH_IE_EXCEPTIONS_RETURN catch (const std::exception&) {
                        return GENERAL_ERROR;
                    }
                    catch (...) {
                        return UNEXPECTED;
                    }
                }();
            }
            callbackToSet(weakThis, statusCode);
        });)
}

InferRequest::operator IInferRequest::Ptr() {
    INFER_REQ_CALL_STATEMENT(return std::make_shared<InferRequestBase>(_impl);)
}

std::vector<VariableState> InferRequest::QueryState() {
    std::vector<VariableState> controller;
    INFER_REQ_CALL_STATEMENT(for (auto&& state
                                  : _impl->QueryState()) {
        controller.emplace_back(VariableState{state, _so});
    })
    return controller;
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

}  // namespace InferenceEngine

namespace ov {

InferRequest::~InferRequest() {
    _impl = {};
}

InferRequest::InferRequest(const std::shared_ptr<ov::IInferRequest>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "InferRequest was not initialized.");
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const Tensor& tensor) {
    OV_INFER_REQ_CALL_STATEMENT({ _impl->set_tensor(port, tensor); });
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
    OV_INFER_REQ_CALL_STATEMENT({ _impl->set_tensors(port, tensors); })
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
        const auto inputs = _impl->get_inputs();
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
        const auto outputs = _impl->get_outputs();
        OPENVINO_ASSERT(outputs.size() == 1,
                        "set_output_tensor() must be called on a function with exactly one parameter.");
        set_tensor(outputs.at(0), tensor);
    });
}

Tensor InferRequest::get_tensor(const ov::Output<const ov::Node>& port) {
    std::vector<std::shared_ptr<void>> soVec;
    OV_INFER_REQ_CALL_STATEMENT({
        // OPENVINO_ASSERT(!_impl->GetBlobs(name),
        //                 "get_tensor shall not be used together with batched "
        //                 "set_tensors/set_input_tensors for name '",
        //                 name,
        //                 "'");
        return _impl->get_tensor(port);
        // soVec = {_so, _impl->getPointerToSo()};
        // Tensor tensor = {blob, soVec};
        // return tensor;
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
        const auto inputs = _impl->get_inputs();
        if (inputs.size() != 1) {
            throw ov::Exception("get_input_tensor() must be called on a function with exactly one parameter.");
        }
        return get_tensor(inputs.at(0));
    });
}

Tensor InferRequest::get_output_tensor() {
    OV_INFER_REQ_CALL_STATEMENT({
        const auto outputs = _impl->get_outputs();
        if (outputs.size() != 1) {
            throw ov::Exception("get_output_tensor() must be called on a function with exactly one parameter.");
        }
        return get_tensor(outputs.at(0));
    });
}

void InferRequest::infer() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->infer();)
}

void InferRequest::cancel() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->cancel();)
}

std::vector<ProfilingInfo> InferRequest::get_profiling_info() const {
    OV_INFER_REQ_CALL_STATEMENT({ return _impl->get_profiling_info(); })
}

void InferRequest::start_async() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->start_async();)
}

void InferRequest::wait() {
    OV_INFER_REQ_CALL_STATEMENT(_impl->wait();)
}

bool InferRequest::wait_for(const std::chrono::milliseconds timeout) {
    OV_INFER_REQ_CALL_STATEMENT(_impl->wait_for(timeout);)
}

void InferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    OV_INFER_REQ_CALL_STATEMENT(_impl->set_callback(std::move(callback));)
}

std::vector<VariableState> InferRequest::query_state() {
    std::vector<VariableState> variable_states;
    std::vector<std::shared_ptr<void>> soVec;
    return _impl->query_state();
    // OV_INFER_REQ_CALL_STATEMENT({
    //     soVec = {_so, _impl->getPointerToSo()};
    //     for (auto&& state : _impl->QueryState()) {
    //         variable_states.emplace_back(VariableState{state, soVec});
    //     }
    // })
    // return variable_states;
}

CompiledModel InferRequest::get_compiled_model() {
    OV_INFER_REQ_CALL_STATEMENT(return {_impl->get_compiled_model(), _so});
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
