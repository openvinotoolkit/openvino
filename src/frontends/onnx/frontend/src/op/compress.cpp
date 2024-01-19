// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/compress.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/squeeze.hpp"
#include "ov_models/ov_builders/reshape.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector compress(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto condition = node.get_ng_inputs().at(1);

    int64_t axis = 0;
    if (node.has_attribute("axis")) {
        axis = node.get_attribute_value<int64_t>("axis");
    } else {
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
    }
    auto axis_node = v0::Constant::create(element::i64, Shape{}, {axis});
    auto zero_node = v0::Constant::create(element::i64, Shape{}, {0});
    auto result =
        std::make_shared<v8::Gather>(data,
                                     std::make_shared<v0::Squeeze>(std::make_shared<v3::NonZero>(condition), zero_node),
                                     axis_node);

    return {result};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
