// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
/// \brief      Function which transforms single ONNX operator to nGraph sub-graph.
OPENVINO_SUPPRESS_DEPRECATED_START
using Operator = std::function<OutputVector(const Node&)>;
OPENVINO_SUPPRESS_DEPRECATED_END

/// \brief      Map which contains ONNX operators accessible by std::string value as a key.
using OperatorSet = std::unordered_map<std::string, Operator>;

}  // namespace onnx_import

}  // namespace ngraph
