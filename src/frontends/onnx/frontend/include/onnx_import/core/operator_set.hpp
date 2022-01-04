// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "onnx_import/core/node.hpp"

namespace ov {
namespace onnx_import {
/// \brief      Function which transforms single ONNX operator to OV sub-graph.
using Operator = std::function<OutputVector(const Node&)>;

/// \brief      Map which contains ONNX operators accessible by std::string value as a key.
using OperatorSet = std::unordered_map<std::string, std::reference_wrapper<const Operator>>;

}  // namespace onnx_import

}  // namespace ov
