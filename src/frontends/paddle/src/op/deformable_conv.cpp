// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "conv2d_utils.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs deformable_conv(const NodeContext& node) {
    auto input = node.get_input("Input");
    auto filter = node.get_input("Filter");
    auto offset = node.get_input("Offset");

    auto strides = node.get_attribute<std::vector<int>>("strides");
    auto dilations = node.get_attribute<std::vector<int>>("dilations");

    auto groups = node.get_attribute<int>("groups");
    auto deformable_groups = node.get_attribute<int>("deformable_groups");

    const auto paddings = get_pads(node);
    const auto pads_begin = paddings.first;
    const auto pads_end = paddings.second;

    const ov::op::PadType auto_pad{ov::op::PadType::EXPLICIT};

    std::shared_ptr<Node> output_node;
    if (node.has_input("Mask")) {
        auto mask = node.get_input("Mask");
        output_node =
            std::make_shared<ov::opset8::DeformableConvolution>(input,
                                                                offset,
                                                                filter,
                                                                mask,
                                                                ov::Strides(strides.begin(), strides.end()),
                                                                pads_begin,
                                                                pads_end,
                                                                ov::Strides(dilations.begin(), dilations.end()),
                                                                auto_pad,
                                                                groups,
                                                                deformable_groups,
                                                                true);
    } else {
        output_node =
            std::make_shared<ov::opset8::DeformableConvolution>(input,
                                                                offset,
                                                                filter,
                                                                ov::Strides(strides.begin(), strides.end()),
                                                                pads_begin,
                                                                pads_end,
                                                                ov::Strides(dilations.begin(), dilations.end()),
                                                                auto_pad,
                                                                groups,
                                                                deformable_groups,
                                                                true);
    }

    return node.default_single_output_mapping({output_node}, {"Output"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
