// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>

#include "ie_remote_context.hpp"

#include "cpp/ie_infer_request.hpp"
#include "cpp/exception2status.hpp"
#include "ie_infer_async_request_base.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

namespace InferenceEngine {

#define INFER_REQ_CALL_STATEMENT(...)                                                              \
    if (_impl == nullptr) IE_THROW(NotAllocated) << "Inference Request is not initialized";        \
    try {                                                                                          \
        __VA_ARGS__                                                                                \
    } catch(...) {details::Rethrow();}

InferRequest::InferRequest(const details::SharedObjectLoader& so,
                           const IInferRequestInternal::Ptr&  impl)
    : _so(so), _impl(impl), actual() {
    IE_ASSERT(_impl != nullptr);
}

IE_SUPPRESS_DEPRECATED_START

InferRequest::InferRequest(IInferRequest::Ptr request,
                           std::shared_ptr<details::SharedObjectLoader> splg)
    : _so(), _impl(), actual(request) {
    if (splg) {
        _so = *splg;
    }

    //  plg can be null, but not the actual
    if (actual == nullptr)
        IE_THROW(NotAllocated) << "InferRequest was not initialized.";
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    if (actual) {
        CALL_STATUS_FNC(SetBlob, name.c_str(), data);
        return;
    }
    INFER_REQ_CALL_STATEMENT(_impl->SetBlob(name, data);)
}

Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    if (actual) {
        Blob::Ptr data;
        CALL_STATUS_FNC(GetBlob, name.c_str(), data);
        std::string error = "Internal error: blob with name `" + name + "` is not allocated!";
        auto blobPtr = data.get();
        const bool remoteBlobPassed = blobPtr->is<RemoteBlob>();
        if (blobPtr == nullptr) IE_THROW() << error;
        if (!remoteBlobPassed && blobPtr->buffer() == nullptr) IE_THROW() << error;
        return data;
    }

    Blob::Ptr blobPtr;
    INFER_REQ_CALL_STATEMENT(blobPtr = _impl->GetBlob(name);)
    std::string error = "Internal error: blob with name `" + name + "` is not allocated!";
    const bool remoteBlobPassed = blobPtr->is<RemoteBlob>();
    if (blobPtr == nullptr) IE_THROW() << error;
    if (!remoteBlobPassed && blobPtr->buffer() == nullptr) IE_THROW() << error;
    return blobPtr;
}

void InferRequest::SetBlob(const std::string &name, const Blob::Ptr &data, const PreProcessInfo& info) {
    if (actual) {
        CALL_STATUS_FNC(SetBlob, name.c_str(), data, info);
        return;
    }

    INFER_REQ_CALL_STATEMENT(_impl->SetBlob(name, data, info);)
}

const PreProcessInfo& InferRequest::GetPreProcess(const std::string& name) const {
    if (actual) {
        const PreProcessInfo* info = nullptr;
        CALL_STATUS_FNC(GetPreProcess, name.c_str(), &info);
        return *info;
    }

    INFER_REQ_CALL_STATEMENT(return _impl->GetPreProcess(name);)
}

void InferRequest::Infer() {
    if (actual) {
        CALL_STATUS_FNC_NO_ARGS(Infer);
        return;
    }

    INFER_REQ_CALL_STATEMENT(_impl->Infer();)
}

void InferRequest::Cancel() {
    if (actual) {
        CALL_STATUS_FNC_NO_ARGS(Cancel);
        return;
    }

    INFER_REQ_CALL_STATEMENT(_impl->Cancel();)
}

std::map<std::string, InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    if (actual) {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        CALL_STATUS_FNC(GetPerformanceCounts, perfMap);
        return perfMap;
    }

    INFER_REQ_CALL_STATEMENT(return _impl->GetPerformanceCounts();)
}

void InferRequest::SetInput(const BlobMap& inputs) {
    if (actual) {
        for (auto&& input : inputs) {
            CALL_STATUS_FNC(SetBlob, input.first.c_str(), input.second);
        }
        return;
    }

    INFER_REQ_CALL_STATEMENT(
        for (auto&& input : inputs) {
            _impl->SetBlob(input.first, input.second);
        }
    )
}

void InferRequest::SetOutput(const BlobMap& results) {
    if (actual) {
        for (auto&& result : results) {
            CALL_STATUS_FNC(SetBlob, result.first.c_str(), result.second);
        }
        return;
    }

    INFER_REQ_CALL_STATEMENT(
        for (auto&& result : results) {
            _impl->SetBlob(result.first, result.second);
        }
    )
}

void InferRequest::SetBatch(const int batch) {
    if (actual) {
        CALL_STATUS_FNC(SetBatch, batch);
        return;
    }

    INFER_REQ_CALL_STATEMENT(_impl->SetBatch(batch);)
}

void InferRequest::StartAsync() {
    if (actual) {
        CALL_STATUS_FNC_NO_ARGS(StartAsync);
        return;
    }

    INFER_REQ_CALL_STATEMENT(_impl->StartAsync();)
}


StatusCode InferRequest::Wait(int64_t millis_timeout) {
    if (actual) {
        ResponseDesc resp;
        if (actual == nullptr) IE_THROW() << "InferRequest was not initialized.";
        auto res = actual->Wait(millis_timeout, &resp);
        if (res != OK && res != RESULT_NOT_READY &&
            res != INFER_NOT_STARTED && res != INFER_CANCELLED) {
            IE_EXCEPTION_SWITCH(res, ExceptionType,
                InferenceEngine::details::ThrowNow<ExceptionType>{}
                    <<= std::stringstream{} << IE_LOCATION << resp.msg)
        }
        return res;
    }

    INFER_REQ_CALL_STATEMENT(return _impl->Wait(millis_timeout);)
}

namespace details {

class ICompletionCallbackWrapper {
public:
    virtual ~ICompletionCallbackWrapper() = default;

    virtual void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept = 0;
};

template <class T>
class CompletionCallbackWrapper : public ICompletionCallbackWrapper {
    T lambda;

public:
    explicit CompletionCallbackWrapper(const T& lambda): lambda(lambda) {}

    void call(InferenceEngine::IInferRequest::Ptr /*request*/, InferenceEngine::StatusCode /*code*/) const
        noexcept override {
        lambda();
    }
};

template <>
class CompletionCallbackWrapper<IInferRequest::CompletionCallback> : public ICompletionCallbackWrapper {
    IInferRequest::CompletionCallback callBack;

public:
    explicit CompletionCallbackWrapper(const IInferRequest::CompletionCallback& callBack): callBack(callBack) {}

    void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept override {
        callBack(request, code);
    }
};

template <>
class CompletionCallbackWrapper<std::function<void(InferRequest, StatusCode)>> : public ICompletionCallbackWrapper {
    std::function<void(InferRequest, StatusCode)> lambda;

public:
    explicit CompletionCallbackWrapper(const std::function<void(InferRequest, InferenceEngine::StatusCode)>& lambda)
        : lambda(lambda) {}

    void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept override {
        lambda(InferRequest(request), code);
    }
};

void callWrapper(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) {
    details::ICompletionCallbackWrapper* pWrapper = nullptr;
    ResponseDesc dsc;
    request->GetUserData(reinterpret_cast<void**>(&pWrapper), &dsc);
    pWrapper->call(request, code);
}

}  // namespace details

void InferRequest::SetCompletionCallbackImpl(std::function<void()> callbackToSet) {
    if (actual) {
        using T = std::function<void()>;
        callback.reset(new details::CompletionCallbackWrapper<T>(callbackToSet));
        CALL_STATUS_FNC(SetUserData, callback.get());
        actual->SetCompletionCallback(InferenceEngine::details::callWrapper);
        return;
    }

    INFER_REQ_CALL_STATEMENT(
        _impl->SetCallback([callbackToSet] (std::exception_ptr) {
            callbackToSet();
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


void InferRequest::SetCompletionCallbackImpl(std::function<void(InferRequest, StatusCode)> callbackToSet) {
    if (actual) {
        using T = std::function<void(InferRequest, StatusCode)>;
        callback.reset(new details::CompletionCallbackWrapper<T>(callbackToSet));
        CALL_STATUS_FNC(SetUserData, callback.get());
        actual->SetCompletionCallback(InferenceEngine::details::callWrapper);
        return;
    }

    INFER_REQ_CALL_STATEMENT(
        auto weakThis = InferRequest{_so, std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*){}}};
        _impl->SetCallback([callbackToSet, weakThis] (std::exception_ptr exceptionPtr) {
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
            callbackToSet(weakThis, statusCode);
        });
    )
}

void InferRequest::SetCompletionCallbackImpl(IInferRequest::CompletionCallback callbackToSet) {
    if (actual) {
        using T = IInferRequest::CompletionCallback;
        callback.reset(new details::CompletionCallbackWrapper<T>(callbackToSet));
        CALL_STATUS_FNC(SetUserData, callback.get());
        actual->SetCompletionCallback(InferenceEngine::details::callWrapper);
        return;
    }

    INFER_REQ_CALL_STATEMENT(
        IInferRequest::Ptr weakThis = InferRequest{_so, std::shared_ptr<IInferRequestInternal>{_impl.get(), [](IInferRequestInternal*){}}};
        _impl->SetCallback([callbackToSet, weakThis] (std::exception_ptr exceptionPtr) {
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
            callbackToSet(weakThis, statusCode);
        });
    )
}

InferRequest::operator IInferRequest::Ptr () {
    if (actual) {
        return actual;
    }

    INFER_REQ_CALL_STATEMENT(
        return std::make_shared<InferRequestBase>(_impl);
    )
}

std::vector<VariableState> InferRequest::QueryState() {
    if (actual) {
        IE_SUPPRESS_DEPRECATED_START
        if (actual == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";
        IVariableState::Ptr pState = nullptr;
        auto res = OK;
        std::vector<VariableState> controller;
        for (size_t idx = 0; res == OK; ++idx) {
            ResponseDesc resp;
            res = actual->QueryState(pState, idx, &resp);
            if (res != OK && res != OUT_OF_BOUNDS) {
                IE_THROW() << resp.msg;
            }
            if (res != OUT_OF_BOUNDS) {
                controller.push_back(VariableState(pState,
                    std::make_shared<details::SharedObjectLoader>(_so)));
            }
        }
        IE_SUPPRESS_DEPRECATED_END

        return controller;
    }

    std::vector<VariableState> controller;
    INFER_REQ_CALL_STATEMENT(
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(VariableState{_so, state});
        }
    )
    return controller;
}

bool InferRequest::operator!() const noexcept {
    return !_impl && !actual;
}

InferRequest::operator bool() const noexcept {
    return (!!_impl) || (!!actual);
}

bool InferRequest::operator!=(const InferRequest& r) const noexcept {
    return !(r == *this);
}

bool InferRequest::operator==(const InferRequest& r) const noexcept {
    return r._impl == _impl && r.actual == actual;
}

}  // namespace InferenceEngine
