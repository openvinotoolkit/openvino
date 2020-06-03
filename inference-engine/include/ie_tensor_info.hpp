// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the TensorInfo structure
 *
 * @file ie_tensor_info.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include <ie_api.h>

namespace InferenceEngine {

/**
 * @deprecated Use ExecutableNetwork::GetExecGraphInfo to get information about an internal graph.
 * This API will be removed in 2021.1 release.
 * @struct TensorInfo
 * @brief This structure describes tensor information
 */
struct INFERENCE_ENGINE_DEPRECATED("Use ExecutableNetwork::GetExecGraphInfo to get information about an internal graph") TensorInfo {
    /**
     * @brief A shared pointer to the TensorInfo object
     */
    IE_SUPPRESS_DEPRECATED_START
    using Ptr = std::shared_ptr<TensorInfo>;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A map of extra info:
     * - memory layout BFYX, BXYF (enum)
     * - size
     * - precision
     */
    std::map<std::string, std::string> extraInfo;
};

}  // namespace InferenceEngine
