// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/and.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector logical_and(const Node& node) {
    return common::handle_opset6_binary_op<default_opset::LogicalAnd>(node);
}
}  // namespace set_1

namespace set_7 {
inline OutputVector logical_and(const Node& node) {
    return {std::make_shared<default_opset::LogicalAnd>(node.get_ng_inputs().at(0), node.get_ng_inputs().at(1))};
}
}  // namespace set_7

}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
