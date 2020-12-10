// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Wrappers from c++ function to c-style one
 * @file exception2status.hpp
 */
#pragma once

#include <string>

#include "description_buffer.hpp"

/**
 * @def THROW_IE_EXCEPTION_WITH_STATUS
 * @brief Throws an exception along with the status (which is converted to the typed exception)
 */
#define THROW_IE_EXCEPTION_WITH_STATUS(__status) THROW_IE_EXCEPTION << \
                        InferenceEngine::details::as_status << InferenceEngine::StatusCode::__status << __status##_str

namespace InferenceEngine {

/**
 * @def TO_STATUS(x)
 * @brief Converts C++ exceptioned function call into a c-style one
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS(x)                                                                                         \
    try {                                                                                                    \
        x;                                                                                                   \
        return OK;                                                                                           \
    } catch (const InferenceEngine::details::InferenceEngineException& iex) {                                \
        return InferenceEngine::DescriptionBuffer((iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR), resp) \
               << iex.what();                                                                                \
    } catch (const InferenceEngine::NotImplemented & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(NOT_IMPLEMENTED, resp) << ex.what();                       \
    } catch (const InferenceEngine::NotFound & ex) {                                                         \
        return InferenceEngine::DescriptionBuffer(NOT_FOUND, resp) << ex.what();                             \
    } catch (const InferenceEngine::OutOfBounds & ex) {                                                      \
        return InferenceEngine::DescriptionBuffer(OUT_OF_BOUNDS, resp) << ex.what();                         \
    } catch (const InferenceEngine::NetworkNotLoaded & ex) {                                                 \
        return InferenceEngine::DescriptionBuffer(NETWORK_NOT_LOADED, resp) << ex.what();                    \
    } catch (const InferenceEngine::ParameterMismatch & ex) {                                                \
        return InferenceEngine::DescriptionBuffer(PARAMETER_MISMATCH, resp) << ex.what();                    \
    } catch (const InferenceEngine::RequestBusy & ex) {                                                      \
        return InferenceEngine::DescriptionBuffer(REQUEST_BUSY, resp) << ex.what();                          \
    } catch (const InferenceEngine::ResultNotReady & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(RESULT_NOT_READY, resp) << ex.what();                      \
    } catch (const InferenceEngine::NotAllocated & ex) {                                                     \
        return InferenceEngine::DescriptionBuffer(NOT_ALLOCATED, resp) << ex.what();                         \
    } catch (const InferenceEngine::InferNotStarted & ex) {                                                  \
        return InferenceEngine::DescriptionBuffer(INFER_NOT_STARTED, resp) << ex.what();                     \
    } catch (const InferenceEngine::NetworkNotRead & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(NETWORK_NOT_READ, resp) << ex.what();                      \
    } catch (const std::exception& ex) {                                                                     \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                         \
    } catch (...) {                                                                                          \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                               \
    }

/**
 * @def TO_STATUS_NO_RESP(x)
 * @brief Converts C++ exceptioned function call into a status code. Does not work with a ResponseDesc object
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS_NO_RESP(x)                                                                                        \
    try {                                                                                                           \
        x;                                                                                                          \
        return OK;                                                                                                  \
    } catch (const InferenceEngine::NotImplemented & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(NOT_IMPLEMENTED) << ex.what();                       \
    } catch (const InferenceEngine::NotFound & ex) {                                                         \
        return InferenceEngine::DescriptionBuffer(NOT_FOUND) << ex.what();                             \
    } catch (const InferenceEngine::OutOfBounds & ex) {                                                      \
        return InferenceEngine::DescriptionBuffer(OUT_OF_BOUNDS) << ex.what();                         \
    } catch (const InferenceEngine::NetworkNotLoaded & ex) {                                                 \
        return InferenceEngine::DescriptionBuffer(NETWORK_NOT_LOADED) << ex.what();                    \
    } catch (const InferenceEngine::ParameterMismatch & ex) {                                                \
        return InferenceEngine::DescriptionBuffer(PARAMETER_MISMATCH) << ex.what();                    \
    } catch (const InferenceEngine::RequestBusy & ex) {                                                      \
        return InferenceEngine::DescriptionBuffer(REQUEST_BUSY) << ex.what();                          \
    } catch (const InferenceEngine::ResultNotReady & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(RESULT_NOT_READY) << ex.what();                      \
    } catch (const InferenceEngine::NotAllocated & ex) {                                                     \
        return InferenceEngine::DescriptionBuffer(NOT_ALLOCATED) << ex.what();                         \
    } catch (const InferenceEngine::InferNotStarted & ex) {                                                  \
        return InferenceEngine::DescriptionBuffer(INFER_NOT_STARTED) << ex.what();                     \
    } catch (const InferenceEngine::NetworkNotRead & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(NETWORK_NOT_READ) << ex.what();                      \
    } catch (const InferenceEngine::details::InferenceEngineException& iex) {                                       \
        return InferenceEngine::DescriptionBuffer(iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR) << iex.what(); \
    } catch (const std::exception& ex) {                                                                            \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR) << ex.what();                                      \
    } catch (...) {                                                                                                 \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                      \
    }

/**
 * @def NO_EXCEPT_CALL_RETURN_STATUS(x)
 * @brief Returns a status code of a called function, handles exeptions and converts to a status code.
 * @ingroup ie_dev_api_error_debug
 */
#define NO_EXCEPT_CALL_RETURN_STATUS(x)                                                                    \
    try {                                                                                                  \
        return x;                                                                                          \
    } catch (const InferenceEngine::NotImplemented & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(NOT_IMPLEMENTED, resp) << ex.what();                       \
    } catch (const InferenceEngine::NotFound & ex) {                                                         \
        return InferenceEngine::DescriptionBuffer(NOT_FOUND, resp) << ex.what();                             \
    } catch (const InferenceEngine::OutOfBounds & ex) {                                                      \
        return InferenceEngine::DescriptionBuffer(OUT_OF_BOUNDS, resp) << ex.what();                         \
    } catch (const InferenceEngine::NetworkNotLoaded & ex) {                                                 \
        return InferenceEngine::DescriptionBuffer(NETWORK_NOT_LOADED, resp) << ex.what();                    \
    } catch (const InferenceEngine::ParameterMismatch & ex) {                                                \
        return InferenceEngine::DescriptionBuffer(PARAMETER_MISMATCH, resp) << ex.what();                    \
    } catch (const InferenceEngine::RequestBusy & ex) {                                                      \
        return InferenceEngine::DescriptionBuffer(REQUEST_BUSY, resp) << ex.what();                          \
    } catch (const InferenceEngine::ResultNotReady & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(RESULT_NOT_READY, resp) << ex.what();                      \
    } catch (const InferenceEngine::NotAllocated & ex) {                                                     \
        return InferenceEngine::DescriptionBuffer(NOT_ALLOCATED, resp) << ex.what();                         \
    } catch (const InferenceEngine::InferNotStarted & ex) {                                                  \
        return InferenceEngine::DescriptionBuffer(INFER_NOT_STARTED, resp) << ex.what();                     \
    } catch (const InferenceEngine::NetworkNotRead & ex) {                                                   \
        return InferenceEngine::DescriptionBuffer(NETWORK_NOT_READ, resp) << ex.what();                      \
    } catch (const InferenceEngine::details::InferenceEngineException& iex) {                              \
        return InferenceEngine::DescriptionBuffer(iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR, resp) \
               << iex.what();                                                                              \
    } catch (const std::exception& ex) {                                                                   \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                       \
    } catch (...) {                                                                                        \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                             \
    }

/**
 * @addtogroup ie_dev_api_error_debug
 * @{
 * @def PARAMETER_MISMATCH_str
 * @brief Defines the `parameter mismatch` message
 */
#define PARAMETER_MISMATCH_str std::string("[PARAMETER_MISMATCH] ")

/**
 * @def NETWORK_NOT_LOADED_str
 * @brief Defines the `network not loaded` message
 */
#define NETWORK_NOT_LOADED_str std::string("[NETWORK_NOT_LOADED] ")

/**
 * @def NOT_FOUND_str
 * @brief Defines the `not found` message
 */
#define NOT_FOUND_str std::string("[NOT_FOUND] ")

/**
 * @def UNEXPECTED_str
 * @brief Defines the `unexpected` message
 */
#define UNEXPECTED_str std::string("[UNEXPECTED] ")

/**
 * @def GENERAL_ERROR_str
 * @brief Defines the `general error` message
 */
#define GENERAL_ERROR_str std::string("[GENERAL ERROR] ")

/**
 * @def RESULT_NOT_READY_str
 * @brief Defines the `result not ready` message
 */
#define RESULT_NOT_READY_str std::string("[RESULT_NOT_READY] ")

/**
 * @def INFER_NOT_STARTED_str
 * @brief Defines the `infer not started` message
 */
#define INFER_NOT_STARTED_str std::string("[INFER_NOT_STARTED] ")

/**
 * @def REQUEST_BUSY_str
 * @brief Defines the `request busy` message
 */
#define REQUEST_BUSY_str std::string("[REQUEST_BUSY] ")

/**
 * @def NOT_IMPLEMENTED_str
 * @brief Defines the `not implemented` message
 */
#define NOT_IMPLEMENTED_str std::string("[NOT_IMPLEMENTED] ")

/**
 * @def NOT_ALLOCATED_str
 * @brief Defines the `not allocated` message
 */
#define NOT_ALLOCATED_str std::string("[NOT_ALLOCATED] ")

/**
 * @}
 */

}  // namespace InferenceEngine
