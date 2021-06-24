// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Wrappers from c++ function to c-style one
 * @file exception2status.hpp
 */
#pragma once

#include <string>

#include "description_buffer.hpp"

namespace InferenceEngine {
#define CATCH_IE_EXCEPTION_TO_STATUS(StatusCode, ExceptionType) catch (const ExceptionType& ex) {   \
    return InferenceEngine::DescriptionBuffer(StatusCode, resp) << ex.what();                       \
}

#define CATCH_IE_EXCEPTIONS_TO_STATUS                                         \
        CATCH_IE_EXCEPTION_TO_STATUS(GENERAL_ERROR, GeneralError)             \
        CATCH_IE_EXCEPTION_TO_STATUS(NOT_IMPLEMENTED, NotImplemented)         \
        CATCH_IE_EXCEPTION_TO_STATUS(NETWORK_NOT_LOADED, NetworkNotLoaded)    \
        CATCH_IE_EXCEPTION_TO_STATUS(PARAMETER_MISMATCH, ParameterMismatch)   \
        CATCH_IE_EXCEPTION_TO_STATUS(NOT_FOUND, NotFound)                     \
        CATCH_IE_EXCEPTION_TO_STATUS(OUT_OF_BOUNDS, OutOfBounds)              \
        CATCH_IE_EXCEPTION_TO_STATUS(UNEXPECTED, Unexpected)                  \
        CATCH_IE_EXCEPTION_TO_STATUS(REQUEST_BUSY, RequestBusy)               \
        CATCH_IE_EXCEPTION_TO_STATUS(RESULT_NOT_READY, ResultNotReady)        \
        CATCH_IE_EXCEPTION_TO_STATUS(NOT_ALLOCATED, NotAllocated)             \
        CATCH_IE_EXCEPTION_TO_STATUS(INFER_NOT_STARTED, InferNotStarted)      \
        CATCH_IE_EXCEPTION_TO_STATUS(NETWORK_NOT_READ, NetworkNotRead)        \
        CATCH_IE_EXCEPTION_TO_STATUS(INFER_CANCELLED, InferCancelled)

/**
 * @def TO_STATUS(x)
 * @brief Converts C++ exceptioned function call into a c-style one
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS(x)                                                                                            \
    try {                                                                                                       \
        x;                                                                                                      \
        return OK;                                                                                              \
    } CATCH_IE_EXCEPTIONS_TO_STATUS catch (const std::exception& ex) {                                          \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                            \
    } catch (...) {                                                                                             \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                  \
    }

#define CALL_STATUS_FNC(function, ...)                                                          \
    if (!actual) IE_THROW() << "Wrapper used was not initialized.";                             \
    ResponseDesc resp;                                                                          \
    auto res = actual->function(__VA_ARGS__, &resp);                                            \
    if (res != OK) IE_EXCEPTION_SWITCH(res, ExceptionType,                                      \
            InferenceEngine::details::ThrowNow<ExceptionType>{}                                 \
                <<= std::stringstream{} << IE_LOCATION << resp.msg)

#define CALL_STATUS_FNC_NO_ARGS(function)                                                                   \
    if (!actual)  IE_THROW() << "Wrapper used in the CALL_STATUS_FNC_NO_ARGS was not initialized.";         \
    ResponseDesc resp;                                                                                      \
    auto res = actual->function(&resp);                                                                     \
    if (res != OK) IE_EXCEPTION_SWITCH(res, ExceptionType,                                                  \
            InferenceEngine::details::ThrowNow<ExceptionType>{}                                             \
                <<= std::stringstream{} << IE_LOCATION)

}  // namespace InferenceEngine
