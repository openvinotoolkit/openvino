// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>

#include "cpp/ie_infer_request.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/base/ie_infer_async_request_base.hpp"
#include "ie_remote_context.hpp"

namespace InferenceEngine {

#define CATCH_IE_EXCEPTION(ExceptionType) catch (const InferenceEngine::ExceptionType& e) {throw e;}

#define CATCH_IE_EXCEPTIONS                     \
        CATCH_IE_EXCEPTION(GeneralError)        \
        CATCH_IE_EXCEPTION(NotImplemented)      \
        CATCH_IE_EXCEPTION(NetworkNotLoaded)    \
        CATCH_IE_EXCEPTION(ParameterMismatch)   \
        CATCH_IE_EXCEPTION(NotFound)            \
        CATCH_IE_EXCEPTION(OutOfBounds)         \
        CATCH_IE_EXCEPTION(Unexpected)          \
        CATCH_IE_EXCEPTION(RequestBusy)         \
        CATCH_IE_EXCEPTION(ResultNotReady)      \
        CATCH_IE_EXCEPTION(NotAllocated)        \
        CATCH_IE_EXCEPTION(InferNotStarted)     \
        CATCH_IE_EXCEPTION(NetworkNotRead)      \
        CATCH_IE_EXCEPTION(InferCancelled)

#define INFER_REQ_CALL_STATEMENT(...)                                                                        \
    if (_impl == nullptr) IE_THROW() << "Inference Requst is not initialized";                     \
    try {                                                                                          \
        __VA_ARGS__                                                                                \
    } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {                                       \
        IE_THROW() << ex.what();                                                                   \
    } catch (...) {                                                                                \
        IE_THROW(Unexpected);                                                                      \
    }

InferRequest::InferRequest(const std::shared_ptr<IInferRequestInternal>&        impl,
                           const std::shared_ptr<details::SharedObjectLoader>&  so) :
    _impl{impl},
    _so{so} {
    if (_impl == nullptr) IE_THROW() << "Inference Requst is not initialized";
}

InferRequest::~InferRequest() {
    _impl = {};
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    INFER_REQ_CALL_STATEMENT(_impl->SetBlob(name, data);)
}

Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    Blob::Ptr blobPtr;
    INFER_REQ_CALL_STATEMENT(blobPtr = _impl->GetBlob(name);)
    std::string error = "Internal error: blob with name `" + name + "` is not allocated!";
    const bool remoteBlobPassed = blobPtr->is<RemoteBlob>();
    if (blobPtr == nullptr) IE_THROW() << error;
    if (!remoteBlobPassed && blobPtr->buffer() == nullptr) IE_THROW() << error;
    return blobPtr;
}

void InferRequest::SetBlob(const std::string &name, const Blob::Ptr &data, const PreProcessInfo& info) {
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
    INFER_REQ_CALL_STATEMENT(
        for (auto&& input : inputs) {
            _impl->SetBlob(input.first, input.second);
        }
    )
}

void InferRequest::SetOutput(const BlobMap& results) {
    INFER_REQ_CALL_STATEMENT(
        for (auto&& result : results) {
            _impl->SetBlob(result.first, result.second);
        }
    )
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

void InferRequest::SetCompletionCallbackImpl(std::function<void()> callback) {
    INFER_REQ_CALL_STATEMENT(
        _impl->SetCallback([callback] (std::exception_ptr) {
            callback();
        });
    )
}


#define CATCH_IE_EXCEPTION_RETURN(StatusCode, ExceptionType) catch (const ExceptionType&) {return StatusCode;}

#define CATCH_IE_EXCEPTIONS_RETURN                                         \
        CATCH_IE_EXCEPTION_RETURN(GENERAL_ERROR, GeneralError)             \
        CATCH_IE_EXCEPTION_RETURN(NOT_IMPLEMENTED, NotImplemented)         \
        CATCH_IE_EXCEPTION_RETURN(NETWORK_NOT_LOADED, NetworkNotLoaded)    \
        CATCH_IE_EXCEPTION_RETURN(PARAMETER_MISMATCH, ParameterMismatch)   \
        CATCH_IE_EXCEPTION_RETURN(NOT_FOUND, NotFound)                     \
        CATCH_IE_EXCEPTION_RETURN(OUT_OF_BOUNDS, OutOfBounds)              \
        CATCH_IE_EXCEPTION_RETURN(UNEXPECTED, Unexpected)                  \
        CATCH_IE_EXCEPTION_RETURN(REQUEST_BUSY, RequestBusy)               \
        CATCH_IE_EXCEPTION_RETURN(RESULT_NOT_READY, ResultNotReady)        \
        CATCH_IE_EXCEPTION_RETURN(NOT_ALLOCATED, NotAllocated)             \
        CATCH_IE_EXCEPTION_RETURN(INFER_NOT_STARTED, InferNotStarted)      \
        CATCH_IE_EXCEPTION_RETURN(NETWORK_NOT_READ, NetworkNotRead)        \
        CATCH_IE_EXCEPTION_RETURN(INFER_CANCELLED, InferCancelled)


void InferRequest::SetCompletionCallbackImpl(std::function<void(InferRequest, StatusCode)> callback) {
    INFER_REQ_CALL_STATEMENT(
        auto weakThis = InferRequest{std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*){}}, _so};
        _impl->SetCallback([callback, weakThis] (std::exception_ptr exceptionPtr) {
            StatusCode statusCode = StatusCode::OK;
            if (exceptionPtr != nullptr) {
                statusCode = [&] {
                    try {
                        std::rethrow_exception(exceptionPtr);
                    } CATCH_IE_EXCEPTIONS_RETURN catch (const std::exception& ex) {
                        return GENERAL_ERROR;
                    } catch (...) {
                        return UNEXPECTED;
                    }
                } ();
            }
            callback(weakThis, statusCode);
        });
    )
}

IE_SUPPRESS_DEPRECATED_START

void InferRequest::SetCompletionCallbackImpl(IInferRequest::CompletionCallback callback) {
    INFER_REQ_CALL_STATEMENT(
        IInferRequest::Ptr weakThis = InferRequest{std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*){}}, _so};
        _impl->SetCallback([callback, weakThis] (std::exception_ptr exceptionPtr) {
            StatusCode statusCode = StatusCode::OK;
            if (exceptionPtr != nullptr) {
                statusCode = [&] {
                    try {
                        std::rethrow_exception(exceptionPtr);
                    } CATCH_IE_EXCEPTIONS_RETURN catch (const std::exception& ex) {
                        return GENERAL_ERROR;
                    } catch (...) {
                        return UNEXPECTED;
                    }
                } ();
            }
            callback(weakThis, statusCode);
        });
    )
}

InferRequest::operator IInferRequest::Ptr () {
    INFER_REQ_CALL_STATEMENT(
        return std::make_shared<InferRequestBase>(_impl);
    )
}

IE_SUPPRESS_DEPRECATED_END

std::vector<VariableState> InferRequest::QueryState() {
    std::vector<VariableState> controller;
    INFER_REQ_CALL_STATEMENT(
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(std::make_shared<VariableStateBase>(state), _so);
        }
    )
    return controller;
}

bool InferRequest::operator!() const noexcept {
    return !_impl;
}

InferRequest::operator bool() const noexcept {
    return !!_impl;
}
}  // namespace InferenceEngine
