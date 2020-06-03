// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for a plugin logging mechanism
 *
 * @file ie_error.hpp
 */
#pragma once

namespace InferenceEngine {
/**
 * @deprecated IErrorListener is not used anymore. An exception is thrown / StatusCode set in case of any unexpected situations
 * The class will be removed in 2021.1 release.
 * @brief This class represents a custom error listener.
 */
class
INFERENCE_ENGINE_DEPRECATED("IErrorListener is not used anymore. An exception is thrown / StatusCode set in case of any unexpected situations")
    IErrorListener {
public:
    /**
     * @brief The plugin calls this method with a null terminated error message (in case of error)
     * @param msg Error message
     */
    virtual void onError(const char* msg) noexcept = 0;
};
}  // namespace InferenceEngine
