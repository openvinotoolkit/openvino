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

#define CALL_STATEMENT(...)                                                                        \
    if (_impl == nullptr) IE_THROW() << "Inference Requst is not initialized";                     \
    try {                                                                                          \
        __VA_ARGS__                                                                                \
    } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {                                       \
        IE_THROW() << ex.what();                                                                   \
    } catch (...) {                                                                                \
        IE_THROW()_WITH_STATUS(Unexpected);                                                        \
    }

InferRequest::InferRequest(const std::shared_ptr<details::SharedObjectLoader>&  plugin,
                           const std::shared_ptr<IInferRequestInternal>&        impl) :
    _impl{impl},
    _plugin{plugin} {
    IE_ASSERT(_impl != nullptr);
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    CALL_STATEMENT(_impl->SetBlob(name, data);)
}

Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    CALL_STATEMENT(auto blobPtr = _impl->GetBlob(name);)
    std::string error = "Internal error: blob with name `" + name + "` is not allocated!";
    const bool remoteBlobPassed = blobPtr->is<RemoteBlob>();
    if (blobPtr == nullptr) IE_THROW() << error;
    if (!remoteBlobPassed && blobPtr->buffer() == nullptr) IE_THROW() << error;
    return blobPtr;
}

void InferRequest::SetBlob(const std::string &name, const Blob::Ptr &data, const PreProcessInfo& info) {
    CALL_STATEMENT(_impl->SetBlob(name.c_str(), data, info);)
}

const PreProcessInfo& InferRequest::GetPreProcess(const std::string& name) const {
    CALL_STATEMENT(return _impl->GetPreProcess(name.c_str());)
}

void InferRequest::Infer() {
    CALL_STATEMENT(_impl->Infer();)
}

void InferRequest::Cancel() {
    CALL_STATEMENT(_impl->Cancel();)
}

std::map<std::string, InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    CALL_STATEMENT(return _impl->GetPerformanceCounts();)
}

void InferRequest::SetInput(const BlobMap& inputs) {
    CALL_STATEMENT(
        for (auto&& input : inputs) {
            _impl->SetBlob(input.first.c_str(), input.second);
        }
    )
}

void InferRequest::SetOutput(const BlobMap& results) {
    CALL_STATEMENT(
        for (auto&& result : results) {
            _impl->SetBlob(result.first.c_str(), result.second);
        }
    )
}

void InferRequest::SetBatch(const int batch) {
    CALL_STATEMENT(_impl->SetBatch(batch);)
}

void InferRequest::StartAsync() {
    CALL_STATEMENT(_impl->StartAsync();)
}


StatusCode InferRequest::Wait(int64_t millis_timeout) {
    CALL_STATEMENT(return _impl->Wait(millis_timeout));
}

void InferRequest::SetCompletionCallback(std::function<void()> callback) {
    CALL_STATEMENT(
        _impl->SetCompletionCallback([callback] (std::exception_ptr) {
            callback();
        });
    )
}

void InferRequest::SetCompletionCallback(std::function<void(InferRequest, StatusCode)> callback) {
    CALL_STATEMENT(
        auto weakThis = InferRequest{_plugin, std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*){}}};
        _impl->SetCompletionCallback([callback, weakThis] (std::exception_ptr exceptionPtr) {
            StatusCode statusCode = StatusCode::OK;
            if (exceptionPtr != nullptr) {
                try {
                    std::rethrow_exception(exceptionPtr);
                } catch (InferenceEngine::details::InferenceEngineException& ieException) {
                    statusCode = ieException.hasStatus() ? ieException.getStatus() : StatusCode::GENERAL_ERROR;
                } catch (...) {
                    statusCode = StatusCode::GENERAL_ERROR;
                }
            }
            callback(weakThis, statusCode);
        });
    )
}

void InferRequest::SetCompletionCallback(IInferRequest::CompletionCallback callback) {
    CALL_STATEMENT(
        IInferRequest::Ptr weakThis = InferRequest{_plugin, std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*){}}};
        _impl->SetCompletionCallback([callback, weakThis] (std::exception_ptr exceptionPtr) {
            StatusCode statusCode = StatusCode::OK;
            if (exceptionPtr != nullptr) {
                try {
                    std::rethrow_exception(exceptionPtr);
                } catch (InferenceEngine::details::InferenceEngineException& ieException) {
                    statusCode = ieException.hasStatus() ? ieException.getStatus() : StatusCode::GENERAL_ERROR;
                } catch (...) {
                    statusCode = StatusCode::GENERAL_ERROR;
                }
            }
            callback(weakThis, statusCode);
        });
    )
}

std::vector<VariableState> InferRequest::QueryState() {
    std::vector<VariableState> controller;
    CALL_STATEMENT(
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(std::make_shared<VariableStateBase>(state));
        }
    )
    return controller;
}

InferRequest::operator IInferRequest::Ptr () {
    CALL_STATEMENT(
        return std::make_shared<InferRequestBase>(_impl);
    )
}

bool InferRequest::operator!() const noexcept {
    return !_impl;
}

InferRequest::operator bool() const noexcept {
    return _impl;
}
}  // namespace InferenceEngine
