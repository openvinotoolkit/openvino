// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector compress(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto condition = node.get_ov_inputs().at(1);

    int64_t axis = 0;
    if (node.has_attribute("axis")) {
        axis = node.get_attribute_value<int64_t>("axis");
    } else {
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
    }
    auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    auto zero_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto result =
        std::make_shared<v8::Gather>(data,
                                     std::make_shared<v0::Squeeze>(std::make_shared<v3::NonZero>(condition), zero_node),
                                     axis_node);

    return {result};
}
ONNX_OP("Compress", OPSET_SINCE(1), ai_onnx::opset_1::compress);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
