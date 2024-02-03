// Copyright (C) 2018-2023 Intel Corporation
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
/// \brief      Function which transforms single ONNX operator to nGraph sub-graph.
OPENVINO_SUPPRESS_DEPRECATED_START
using Operator = std::function<ov::OutputVector(const Node&)>;
OPENVINO_SUPPRESS_DEPRECATED_END

/// \brief      Map which contains ONNX operators accessible by std::string value as a key.
using OperatorSet = std::unordered_map<std::string, Operator>;

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
