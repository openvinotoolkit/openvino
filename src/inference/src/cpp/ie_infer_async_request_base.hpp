// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp/exception2status.hpp"
#include "cpp_interfaces/plugin_itt.hpp"
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include "ie_iinfer_request.hpp"
#include "ie_preprocess.hpp"

namespace InferenceEngine {

#define CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(StatusCode, ExceptionType)     \
    catch (const InferenceEngine::ExceptionType& ex) {                      \
        return InferenceEngine::DescriptionBuffer(StatusCode) << ex.what(); \
    }

#define CATCH_OV_EXCEPTION_TO_STATUS_NO_RESP(StatusCode, ExceptionType)     \
    catch (const ov::ExceptionType& ex) {                                   \
        return InferenceEngine::DescriptionBuffer(StatusCode) << ex.what(); \
    }

#define CATCH_IE_EXCEPTIONS_TO_STATUS_NO_RESP                                   \
    CATCH_OV_EXCEPTION_TO_STATUS_NO_RESP(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_OV_EXCEPTION_TO_STATUS_NO_RESP(GENERAL_ERROR, Exception)              \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(GENERAL_ERROR, GeneralError)           \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(NETWORK_NOT_LOADED, NetworkNotLoaded)  \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(PARAMETER_MISMATCH, ParameterMismatch) \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(NOT_FOUND, NotFound)                   \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(OUT_OF_BOUNDS, OutOfBounds)            \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(UNEXPECTED, Unexpected)                \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(REQUEST_BUSY, RequestBusy)             \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(RESULT_NOT_READY, ResultNotReady)      \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(NOT_ALLOCATED, NotAllocated)           \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(INFER_NOT_STARTED, InferNotStarted)    \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(NETWORK_NOT_READ, NetworkNotRead)      \
    CATCH_IE_EXCEPTION_TO_STATUS_NO_RESP(INFER_CANCELLED, InferCancelled)

/**
 * @def TO_STATUS_NO_RESP(x)
 * @brief Converts C++ exceptioned function call into a status code. Does not work with a ResponseDesc object
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS_NO_RESP(x)                                                                                        \
    try {                                                                                                           \
        x;                                                                                                          \
        return OK;                                                                                                  \
    } CATCH_IE_EXCEPTIONS_TO_STATUS_NO_RESP catch (const std::exception& ex) {                                      \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR) << ex.what();                                      \
    } catch (...) {                                                                                                 \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                      \
    }

#define CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(StatusCode, ExceptionType)        \
catch (const InferenceEngine::ExceptionType& ex) {                              \
    return InferenceEngine::DescriptionBuffer(StatusCode, resp) << ex.what();   \
}

#define CATCH_OV_EXCEPTION_CALL_RETURN_STATUS(StatusCode, ExceptionType)        \
catch (const ov::ExceptionType& ex) {                              \
    return InferenceEngine::DescriptionBuffer(StatusCode, resp) << ex.what();   \
}

#define CATCH_IE_EXCEPTIONS_CALL_RETURN_STATUS                                         \
        CATCH_OV_EXCEPTION_CALL_RETURN_STATUS(NOT_IMPLEMENTED, NotImplemented)         \
        CATCH_OV_EXCEPTION_CALL_RETURN_STATUS(GENERAL_ERROR, Exception)                \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(GENERAL_ERROR, GeneralError)             \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(NOT_IMPLEMENTED, NotImplemented)         \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(NETWORK_NOT_LOADED, NetworkNotLoaded)    \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(PARAMETER_MISMATCH, ParameterMismatch)   \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(NOT_FOUND, NotFound)                     \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(OUT_OF_BOUNDS, OutOfBounds)              \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(UNEXPECTED, Unexpected)                  \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(REQUEST_BUSY, RequestBusy)               \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(RESULT_NOT_READY, ResultNotReady)        \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(NOT_ALLOCATED, NotAllocated)             \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(INFER_NOT_STARTED, InferNotStarted)      \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(NETWORK_NOT_READ, NetworkNotRead)        \
        CATCH_IE_EXCEPTION_CALL_RETURN_STATUS(INFER_CANCELLED, InferCancelled)

/**
 * @def NO_EXCEPT_CALL_RETURN_STATUS(x)
 * @brief Returns a status code of a called function, handles exeptions and converts to a status code.
 * @ingroup ie_dev_api_error_debug
 */
#define NO_EXCEPT_CALL_RETURN_STATUS(x)                                                                         \
    try {                                                                                                       \
        return x;                                                                                               \
    } CATCH_IE_EXCEPTIONS_CALL_RETURN_STATUS catch (const std::exception& ex) {                                 \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                            \
    } catch (...) {                                                                                             \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                  \
    }

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief Inference request `noexcept` wrapper which accepts IInferRequestInternal derived instance which can throw exceptions
 * @ingroup ie_dev_api_infer_request_api
 */
class InferRequestBase : public IInferRequest {
    std::shared_ptr<IInferRequestInternal> _impl;

public:
    /**
     * @brief Constructor with actual underlying implementation.
     * @param impl Underlying implementation of type IInferRequestInternal
     */
    explicit InferRequestBase(std::shared_ptr<IInferRequestInternal> impl): _impl(impl) {}

    StatusCode Infer(ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "Infer");
        TO_STATUS(_impl->Infer());
    }

    StatusCode Cancel(ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "Cancel");
        TO_STATUS(_impl->Cancel());
    }

    StatusCode GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap,
                                    ResponseDesc* resp) const noexcept override {
        TO_STATUS(perfMap = _impl->GetPerformanceCounts());
    }

    StatusCode SetBlob(const char* name, const Blob::Ptr& data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetBlob(name, data));
    }

    StatusCode GetBlob(const char* name, Blob::Ptr& data, ResponseDesc* resp) noexcept override {
        TO_STATUS(data = _impl->GetBlob(name));
    }

    StatusCode GetPreProcess(const char* name, const PreProcessInfo** info, ResponseDesc *resp) const noexcept override {
        TO_STATUS(*info = &(_impl->GetPreProcess(name)));
    }

    StatusCode StartAsync(ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "StartAsync");
        TO_STATUS(_impl->StartAsync());
    }

    StatusCode Wait(int64_t millis_timeout, ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "Wait");
        NO_EXCEPT_CALL_RETURN_STATUS(_impl->Wait(millis_timeout));
    }

    StatusCode SetCompletionCallback(CompletionCallback callback) noexcept override {
        auto weakImpl = std::shared_ptr<IInferRequestInternal>(_impl.get(), [](IInferRequestInternal*){});
        TO_STATUS_NO_RESP(_impl->SetCallback([callback, weakImpl] (std::exception_ptr exceptionPtr) {
            StatusCode statusCode = [&] ()-> StatusCode {
                if (exceptionPtr) {
                    TO_STATUS_NO_RESP(std::rethrow_exception(exceptionPtr));
                } else {
                    return OK;
                }
            } ();
            callback(std::make_shared<InferRequestBase>(weakImpl), statusCode);
        }));
    }

    StatusCode GetUserData(void** data, ResponseDesc* resp) noexcept override {
        if (data != nullptr) {
            TO_STATUS(*data = _impl->GetUserData());
        } else {
            return GENERAL_ERROR;
        }
    }

    StatusCode SetUserData(void* data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetUserData(data));
    }
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
