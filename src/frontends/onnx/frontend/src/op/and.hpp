// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/and.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector logical_and(const Node& node) {
    CHECK_VALID_NODE(node,
                    !node.has_attribute("axis") && !node.has_attribute("broadcast"),
                    "Legacy broadcast mode of And operator is not supported");
    return {std::make_shared<default_opset::LogicalAnd>(node.get_ng_inputs().at(0), node.get_ng_inputs().at(1))};
}
namespace set_7 {
inline OutputVector logical_and(const Node& node) {
    using set_7::logical_and;
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
