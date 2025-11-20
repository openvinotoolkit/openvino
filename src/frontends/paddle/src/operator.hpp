// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ov {
namespace frontend {
namespace paddle {

/**
 * @brief Structure to hold PaddlePaddle operator information from JSON format models
 *
 * This structure represents a single operator in the PaddlePaddle JSON model format (PP-OCRv5).
 * It contains the operator type, input/output tensor names, and attributes.
 */
struct Operator {
    std::string type;                  ///< Operator type (e.g., "conv2d", "pool2d", "relu")
    std::vector<std::string> inputs;   ///< Input tensor names
    std::vector<std::string> outputs;  ///< Output tensor names
    nlohmann::json attributes;         ///< Operator attributes as JSON object
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov