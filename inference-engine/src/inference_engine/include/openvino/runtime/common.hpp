// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime common aliases that depend only from external API
 *
 * @file openvino/runtime/common.hpp
 */
#pragma once

#include <map>
#include <string>

namespace ov {
namespace ie = InferenceEngine;
namespace runtime {
/**
 * @brief This type of map is commonly used to pass set of parameters
 */
using ConfigMap = std::map<std::string, std::string>;
}  // namespace runtime
}  // namespace ov