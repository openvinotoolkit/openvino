// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_infer_request.hpp"

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "dev/converter_utils.hpp"
#include "ie_infer_async_request_base.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_context.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

#define INFER_REQ_CALL_STATEMENT(...)                                     \
    if (_impl == nullptr)                                                 \
        IE_THROW(NotAllocated) << "Inference Request is not initialized"; \
    try {                                                                 \
        __VA_ARGS__                                                       \
    } catch (...) {                                                       \
        ::InferenceEngine::details::Rethrow();                            \
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

#define CATCH_OV_EXCEPTION_RETURN(StatusCode, ExceptionType) \
    catch (const ::ov::ExceptionType&) {                     \
        return StatusCode;                                   \
    }

#define CATCH_IE_EXCEPTIONS_RETURN                                   \
    CATCH_OV_EXCEPTION_RETURN(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_OV_EXCEPTION_RETURN(GENERAL_ERROR, Exception)              \
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
