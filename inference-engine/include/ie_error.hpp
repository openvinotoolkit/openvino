// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for a plugin logging mechanism
 * @file ie_error.hpp
 */
#pragma once

namespace InferenceEngine {
/**
 * @brief This class represents a custom error listener.
 * Plugin consumers can provide it via InferenceEngine::SetLogCallback
 */
class IErrorListener {
public:
    /**
     * @brief The plugin calls this method with a null terminated error message (in case of error)
     * @param msg Error message
     */
    virtual void onError(const char *msg) noexcept = 0;
};
}  // namespace InferenceEngine
