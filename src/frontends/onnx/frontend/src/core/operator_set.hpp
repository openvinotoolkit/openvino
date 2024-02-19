// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
/// \brief      Function which transforms single ONNX operator to OV sub-graph.

using Operator = std::function<OutputVector(const Node&)>;

/// \brief      Map which contains ONNX operators accessible by std::string value as a key.
using OperatorSet = std::unordered_map<std::string, Operator>;

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
