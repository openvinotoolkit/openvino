// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the PrimitiveInfo struct
 *
 * @file ie_primitive_info.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_tensor_info.hpp"

namespace InferenceEngine {

/**
 * @deprecated Use ExecutableNetwork::GetExecGraphInfo to get information about an internal graph.
 * @brief Structure with information about Primitive
 */
struct INFERENCE_ENGINE_DEPRECATED("Use ExecutableNetwork::GetExecGraphInfo to get information about an internal graph") PrimitiveInfo {
    /**
     * @brief A shared pointer to PrimitiveInfo object
     */
    IE_SUPPRESS_DEPRECATED_START
    using Ptr = std::shared_ptr<PrimitiveInfo>;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Some internal id, could be used as a name
     */
    std::string sId;

    /**
     * @brief Implementation type of this kernel
     */
    std::string sType;

    /**
     * @brief Mainly the allocation of the output tensor
     */
    int iPreAllocatedMemory;

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Vector of TensorInfo objects that are related to input tensors
     */
    std::vector<TensorInfo::Ptr> inputs;

    /**
     * @brief Vector of TensorInfo object that are related to outputs tensors
     */
    std::vector<TensorInfo::Ptr> outputs;

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Any other important textual information user might find interesting about this kernel
     */
    std::map<std::string, std::string> extraInfo;
};

}  // namespace InferenceEngine
