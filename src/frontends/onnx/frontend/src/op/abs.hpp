// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector abs(const Node& node) {
    CHECK_VALID_NODE(node,
                     !node.has_attribute("consumed_inputs"),
                     "consumed_inputs legacy attribute of Abs op is not supported");
    return {std::make_shared<default_opset::Abs>(node.get_ng_inputs().at(0))};
}
}  // namespace set_1

namespace set_6 {
using set_1::abs;
}  // namespace set_6

namespace set_13 {
using set_6::abs;
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
