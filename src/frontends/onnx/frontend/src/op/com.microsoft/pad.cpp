// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/pad.hpp"

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector pad(const Node& node) {
    auto inputs = node.get_ng_inputs();
    auto data = inputs.at(0);
    auto pads = inputs.at(1);

    std::string pad_mode = node.get_attribute_value<std::string>("mode", "constant");
    ov::op::PadMode mode;
    ov::Output<ov::Node> pad_value;
    if (pad_mode == "constant") {
        mode = ov::op::PadMode::CONSTANT;
        if (inputs.size() > 2) {
            pad_value = inputs.at(2);
        } else {
            pad_value = default_opset::Constant::create(data.get_element_type(), Shape{}, {0});
        }
    } else if (pad_mode == "edge") {
        mode = ov::op::PadMode::EDGE;
    } else if (pad_mode == "reflect") {
        mode = ov::op::PadMode::REFLECT;
    } else {
        throw ngraph_error("Unsupported pad_mode in ONNX com.microsoft.Pad operator");
    }

    auto pads_shape = pads.get_shape();
    if (!((pads_shape.size() == 1 && pads_shape[0] == 2 * data.get_shape().size()) ||
          (pads_shape.size() == 2 && pads_shape[0] == 1 && pads_shape[1] == 2 * data.get_shape().size()))) {
        throw ngraph_error("Invalid pads tensor shape in ONNX Pad operator");
    }

    auto pads_begin = std::make_shared<default_opset::Slice>(
        pads,
        default_opset::Constant::create(element::i64, Shape{}, {0}),
        default_opset::Constant::create(element::i64, Shape{}, {pads_shape[0] / 2}),
        default_opset::Constant::create(element::i64, Shape{}, {1}));

    auto pads_end = std::make_shared<default_opset::Slice>(
        pads,
        default_opset::Constant::create(element::i64, Shape{}, {pads_shape[0] / 2}),
        default_opset::Constant::create(element::i64, Shape{}, {pads_shape[0]}),
        default_opset::Constant::create(element::i64, Shape{}, {1}));

    if (pad_mode == "constant") {
        return {std::make_shared<default_opset::Pad>(data, pads_begin, pads_end, pad_value, mode)};
    } else {
        return {std::make_shared<default_opset::Pad>(data, pads_begin, pads_end, mode)};
    }
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
