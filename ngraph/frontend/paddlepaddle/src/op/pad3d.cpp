// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad3d.hpp"
#include <ngraph/opsets/opset6.hpp>
#include <paddlepaddle_frontend/exceptions.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

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

            // TODO: Only functional support Int padding format, further verify in #55169
            if (node.has_attribute<std::vector<int32_t>>("paddings"))
            {
                auto paddings_vector = node.get_attribute<std::vector<int32_t>>("paddings");
                PDPD_ASSERT(paddings_vector.size() == 6,
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
                throw ngraph_error("Unsupported paddings attribute!");
            }

            auto pads_begin = std::vector<int32_t>(5, 0);
            auto pads_end = std::vector<int32_t>(5, 0);

            Output<Node> values;
            Output<Node> padding_begin;
            Output<Node> padding_end;

            op::PadMode pad_mode;
            // TODO Support Circular mode in future #55169
            if (mode == "constant")
            {
                pad_mode = op::PadMode::CONSTANT;
                values = opset6::Constant::create(element::f32, Shape{}, {value});
            }
            else if (mode == "reflect")
            {
                pad_mode = op::PadMode::REFLECT;
            }
            else if (mode == "replicate")
            {
                pad_mode = op::PadMode::EDGE;
            }
            else
            {
                throw ngraph_error("Unsupported 3d paddings mode: [" + mode + "]");
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
                throw ngraph_error("Unsupported 3d paddings data_format: [" + data_format + "]");
            }

            padding_begin =
                opset6::Constant::create(element::i32, Shape{pads_begin.size()}, pads_begin);
            padding_end = opset6::Constant::create(element::i32, Shape{pads_end.size()}, pads_end);

            if (mode == "constant")
                return node.default_single_output_mapping(
                    {std::make_shared<opset6::Pad>(
                        data, padding_begin, padding_end, values, pad_mode)},
                    {"Out"});
            else
                return node.default_single_output_mapping(
                    {std::make_shared<opset6::Pad>(data, padding_begin, padding_end, pad_mode)},
                    {"Out"});
        }
    } // namespace op
} // namespace pdpd
