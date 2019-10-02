// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the TensorInfo structure
 * @file ie_tensor_info.hpp
 */

#pragma once

#include <string>
#include <memory>
#include <map>

namespace InferenceEngine {

/**
* @struct TensorInfo
* @brief This structure describes tensor information
*/
struct TensorInfo {
    /**
    * @brief A shared pointer to the TensorInfo object
    */
    using Ptr = std::shared_ptr<TensorInfo>;

    /**
    * @brief A map of extra info:
    * - memory layout BFYX, BXYF (enum)
    * - size
    * - precision
    */
    std::map<std::string, std::string> extraInfo;
};

}  // namespace InferenceEngine
