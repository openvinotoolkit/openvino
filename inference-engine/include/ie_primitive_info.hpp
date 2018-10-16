// Copyright (C) 2018 Intel Corporation
//
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

struct PrimitiveInfo {
    using Ptr = std::shared_ptr<PrimitiveInfo>;

    std::string sId;          // some internal id, could be used as a name
    std::string sType;        // implementation type of this kernel
    int iPreAllocatedMemory;  // mainly the allocation of the output tensor

    std::vector<TensorInfo::Ptr> inputs;
    std::vector<TensorInfo::Ptr> outputs;

    std::map<std::string, std::string> extraInfo;  // any other important textual information user might find interesting about this kernel
};

}  // namespace InferenceEngine
