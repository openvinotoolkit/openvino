// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs pad3d(const NodeContext& node) {
    auto data = node.get_input("X");
    auto mode = node.get_attribute<std::string>("mode");
    auto value = node.get_attribute<float>("value", 0.0);
    auto data_format = node.get_attribute<std::string>("data_format");

    auto paddings = std::vector<int32_t>(6, 0);

    // padding of type int feature only supported by paddle 'develop'
    // version(>=2.1.0)
    if (node.has_attribute("paddings")) {
        auto paddings_vector = node.get_attribute<std::vector<int32_t>>("paddings");
        PADDLE_OP_CHECK(node, paddings_vector.size() == 6, "paddings Params size should be 6 in pad3d!");
        paddings = paddings_vector;
    } else if (node.has_attribute("paddings")) {
        auto padding_int = node.get_attribute<int32_t>("paddings");
        for (int i = 0; i < 6; i++)
            paddings[i] = padding_int;
    } else {
        PADDLE_OP_CHECK(node, false, "Unsupported paddings attribute!");
    }

    auto pads_begin = std::vector<int32_t>(5, 0);
    auto pads_end = std::vector<int32_t>(5, 0);

    Output<ov::Node> values;
    Output<ov::Node> padding_begin;
    Output<ov::Node> padding_end;

    ov::op::PadMode pad_mode;
    // TODO Support Circular mode in #55704
    if (mode == "constant") {
        pad_mode = ov::op::PadMode::CONSTANT;
        values = ov::opset6::Constant::create(element::f32, ov::Shape{}, {value});
    } else if (mode == "reflect") {
        pad_mode = ov::op::PadMode::REFLECT;
    } else if (mode == "replicate") {
        pad_mode = ov::op::PadMode::EDGE;
    } else {
        PADDLE_OP_CHECK(node, false, "Unsupported 3d paddings mode: [" + mode + "]");
    }

    if (data_format == "NCDHW") {
        pads_begin[4] = paddings[0];  // left
        pads_end[4] = paddings[1];    // right
        pads_begin[3] = paddings[2];  // top
        pads_end[3] = paddings[3];    // down
        pads_begin[2] = paddings[4];  // front
        pads_end[2] = paddings[5];    // back
    } else if (data_format == "NDHWC") {
        pads_begin[3] = paddings[0];  // left
        pads_end[3] = paddings[1];    // right
        pads_begin[2] = paddings[2];  // top
        pads_end[2] = paddings[3];    // down
        pads_begin[1] = paddings[4];  // front
        pads_end[1] = paddings[5];    // back
    } else {
        PADDLE_OP_CHECK(node, false, "Unsupported 3d paddings data_format: [" + data_format + "]");
    }

    padding_begin = ov::opset6::Constant::create(element::i32, ov::Shape{pads_begin.size()}, pads_begin);
    padding_end = ov::opset6::Constant::create(element::i32, ov::Shape{pads_end.size()}, pads_end);

    if (mode == "constant")
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset6::Pad>(data, padding_begin, padding_end, values, pad_mode)},
            {"Out"});
    else
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset6::Pad>(data, padding_begin, padding_end, pad_mode)},
            {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
