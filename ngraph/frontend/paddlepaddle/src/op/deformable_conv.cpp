// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_conv.hpp"
#include <ngraph/opsets/opset8.hpp>
#include "conv2d_utils.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs deformable_conv(const NodeContext& node)
                {
                    auto input = node.get_ng_input("Input");
                    auto filter = node.get_ng_input("Filter");
                    auto offset = node.get_ng_input("Offset");

                    auto strides = node.get_attribute<std::vector<int>>("strides");
                    auto dilations = node.get_attribute<std::vector<int>>("dilations");

                    auto groups = node.get_attribute<int>("groups");
                    auto deformable_groups = node.get_attribute<int>("deformable_groups");
                    // auto im2col_step = node.get_attribute<int>("im2col_step"); // TODO

                    const auto paddings = get_pads(node);
                    const auto pads_begin = paddings.first;
                    const auto pads_end = paddings.second;

                    const ngraph::op::PadType auto_pad{ngraph::op::PadType::EXPLICIT};

                    std::shared_ptr<Node> output_node;
                    if (node.has_ng_input("Mask"))
                    {
                        auto mask = node.get_ng_input("Mask");
                        output_node = std::make_shared<ngraph::opset8::DeformableConvolution>(
                            input,
                            offset,
                            filter,
                            mask,
                            ngraph::Strides(strides.begin(), strides.end()),
                            pads_begin,
                            pads_end,
                            ngraph::Strides(dilations.begin(), dilations.end()),
                            auto_pad,
                            groups,
                            deformable_groups,
                            true);
                    }
                    else
                    {
                        output_node = std::make_shared<ngraph::opset8::DeformableConvolution>(
                            input,
                            offset,
                            filter,
                            ngraph::Strides(strides.begin(), strides.end()),
                            pads_begin,
                            pads_end,
                            ngraph::Strides(dilations.begin(), dilations.end()),
                            auto_pad,
                            groups,
                            deformable_groups,
                            true);
                    }

                    return node.default_single_output_mapping({output_node}, {"Output"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph