// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/reciprocal.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"

using namespace ov::op;
using ov::Shape;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector reciprocal(const Node& node) {
    auto data = node.get_ng_inputs().at(0);

    auto one_node = v0::Constant::create(data.get_element_type(), Shape{}, {1});
    return {std::make_shared<v1::Divide>(one_node, data)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
