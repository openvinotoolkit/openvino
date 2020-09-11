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
    } catch (const std::exception& ex) {                                                                     \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                         \
    } catch (...) {                                                                                          \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                               \
    }

/**
 * @def TO_STATUSVAR(x, statusVar, descBufferVar)
 * @brief Converts C++ exceptioned function call to a status variable
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUSVAR(x, statusVar, descBufferVar)                                                                      \
    do {                                                                                                               \
        try {                                                                                                          \
            x;                                                                                                         \
            statusVar = OK;                                                                                            \
        } catch (const InferenceEngine::details::InferenceEngineException& iex) {                                      \
            statusVar =                                                                                                \
                InferenceEngine::DescriptionBuffer((iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR), descBufferVar) \
                << iex.what();                                                                                         \
        } catch (const std::exception& ex) {                                                                           \
            statusVar = InferenceEngine::DescriptionBuffer(GENERAL_ERROR, descBufferVar) << ex.what();                 \
        } catch (...) {                                                                                                \
            statusVar = InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                \
        }                                                                                                              \
    } while (false)

/**
 * @def TO_STATUS_NO_RESP(x)
 * @brief Converts C++ exceptioned function call into a status code. Does not work with a ResponseDesc object
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS_NO_RESP(x)                                                                                        \
    try {                                                                                                           \
        x;                                                                                                          \
        return OK;                                                                                                  \
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
