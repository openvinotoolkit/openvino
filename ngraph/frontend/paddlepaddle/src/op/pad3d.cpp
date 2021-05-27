// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad3d.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs pad3d(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto mode = node.get_attribute<std::string>("mode");
                    auto value = node.get_attribute<float>("value", 0.0);
                    auto data_format = node.get_attribute<std::string>("data_format");

                    auto paddings = std::vector<int32_t>(6, 0);

                    // padding of type int feature only supported by PaddlePaddle 'develop'
                    // version(>=2.1.0)
                    if (node.has_attribute<std::vector<int32_t>>("paddings"))
                    {
                        auto paddings_vector = node.get_attribute<std::vector<int32_t>>("paddings");
                        PDPD_OP_VALIDATION_CHECK(node,
                                                 paddings_vector.size() == 6,
                                                 "paddings Params size should be 6 in pad3d!");
                        paddings = paddings_vector;
                    }
                    else if (node.has_attribute<int32_t>("paddings"))
                    {
                        auto padding_int = node.get_attribute<int32_t>("paddings");
                        for (int i = 0; i < 6; i++)
                            paddings[i] = padding_int;
                    }
                    else
                    {
                        throw ngraph::ngraph_error("Unsupported paddings attribute!");
                    }

                    auto pads_begin = std::vector<int32_t>(5, 0);
                    auto pads_end = std::vector<int32_t>(5, 0);

                    Output<ngraph::Node> values;
                    Output<ngraph::Node> padding_begin;
                    Output<ngraph::Node> padding_end;

                    ngraph::op::PadMode pad_mode;
                    // TODO Support Circular mode in #55704
                    if (mode == "constant")
                    {
                        pad_mode = ngraph::op::PadMode::CONSTANT;
                        values = ngraph::opset6::Constant::create(
                            element::f32, ngraph::Shape{}, {value});
                    }
                    else if (mode == "reflect")
                    {
                        pad_mode = ngraph::op::PadMode::REFLECT;
                    }
                    else if (mode == "replicate")
                    {
                        pad_mode = ngraph::op::PadMode::EDGE;
                    }
                    else
                    {
                        throw ngraph::ngraph_error("Unsupported 3d paddings mode: [" + mode + "]");
                    }

                    if (data_format == "NCDHW")
                    {
                        pads_begin[4] = paddings[0]; // left
                        pads_end[4] = paddings[1];   // right
                        pads_begin[3] = paddings[2]; // top
                        pads_end[3] = paddings[3];   // down
                        pads_begin[2] = paddings[4]; // front
                        pads_end[2] = paddings[5];   // back
                    }
                    else if (data_format == "NDHWC")
                    {
                        pads_begin[3] = paddings[0]; // left
                        pads_end[3] = paddings[1];   // right
                        pads_begin[2] = paddings[2]; // top
                        pads_end[2] = paddings[3];   // down
                        pads_begin[1] = paddings[4]; // front
                        pads_end[1] = paddings[5];   // back
                    }
                    else
                    {
                        throw ngraph::ngraph_error("Unsupported 3d paddings data_format: [" +
                                                   data_format + "]");
                    }

                    padding_begin = ngraph::opset6::Constant::create(
                        element::i32, ngraph::Shape{pads_begin.size()}, pads_begin);
                    padding_end = ngraph::opset6::Constant::create(
                        element::i32, ngraph::Shape{pads_end.size()}, pads_end);

                    if (mode == "constant")
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Pad>(
                                data, padding_begin, padding_end, values, pad_mode)},
                            {"Out"});
                    else
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Pad>(
                                data, padding_begin, padding_end, pad_mode)},
                            {"Out"});
                }
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
