// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the PrimitiveInfo struct
 * @file ie_primitive_info.hpp
 */

#pragma once

#include "ie_tensor_info.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace InferenceEngine {

/**
* @brief Structure with information about Primitive
*/
struct PrimitiveInfo {
    /**
    * @brief A shared pointer to PrimitiveInfo object
    */
    using Ptr = std::shared_ptr<PrimitiveInfo>;

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

    /**
    * @brief Vector of TensorInfo objects that are related to input tensors
    */
    std::vector<TensorInfo::Ptr> inputs;

    /**
    * @brief Vector of TensorInfo object that are related to outputs tensors
    */
    std::vector<TensorInfo::Ptr> outputs;

    /**
    * @brief Any other important textual information user might find interesting about this kernel
    */
    std::map<std::string, std::string> extraInfo;
};

}  // namespace InferenceEngine
