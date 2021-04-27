//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "pad3d.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs pad3d (const NodeContext& node) {
    // TODO
    auto data = node.get_ng_input("X");
    auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
    auto mode = node.get_attribute<std::string>("mode");
    auto value = node.get_attribute<float>("value");
    auto data_format = node.get_attribute<std::string>("data_format");

    auto pads_begin = std::vector<int32_t>(5, 0);
    auto pads_end = std::vector<int32_t>(5, 0);
//    auto value_v = std::vector<float>(1, value);

    Output<ngraph::Node> values;
    Output<ngraph::Node> padding_begin;
    Output<ngraph::Node> padding_end;

    if (paddings.size() != 6)
        throw ngraph::ngraph_error("paddings Params size should be 6 in pad3d!");

    ngraph::op::PadMode pad_mode;

    if (mode == "constant") {
        pad_mode = ngraph::op::PadMode::CONSTANT;
        values = ngraph::opset6::Constant::create(
                     element::f32, ngraph::Shape{}, {value});
    } else if (mode == "reflect") {
        pad_mode = ngraph::op::PadMode::REFLECT;
    } else if (mode == "replicate") {
        pad_mode = ngraph::op::PadMode::EDGE;
    } else {
        throw ngraph::ngraph_error("Unsupported 3d paddings mode: [" + mode + "]");
    }

    if (data_format == "NCDHW") {
        pads_begin[4] = paddings[0]; //left
        pads_end[4] = paddings[1]; //right
        pads_begin[3] = paddings[2]; //top
        pads_end[3] = paddings[3]; //down
        pads_begin[2] = paddings[4]; //front
        pads_end[2] = paddings[5]; //back

    } else if (data_format == "NDHWC") {
        pads_begin[3] = paddings[0]; //left
        pads_end[3] = paddings[1]; //right
        pads_begin[2] = paddings[2]; //top
        pads_end[2] = paddings[3]; //down
        pads_begin[1] = paddings[4]; //front
        pads_end[1] = paddings[5]; //back

    } else {
        throw ngraph::ngraph_error("Unsupported 3d paddings data_format: [" + data_format + "]");
    }

    padding_begin = ngraph::opset6::Constant::create(
                        element::i32, ngraph::Shape{5}, pads_begin);
    padding_end = ngraph::opset6::Constant::create(
                      element::i32, ngraph::Shape{5}, pads_end);

    if (mode == "constant")
//        return {std::make_shared<ngraph::opset6::Pad>(data, padding_begin, padding_end, values, pad_mode)};
        return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Pad>(data, padding_begin, padding_end, values, pad_mode)}, {"Out"});
    else
//        return {std::make_shared<ngraph::opset6::Pad>(data, padding_begin, padding_end, pad_mode)};
        return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Pad>(data, padding_begin, padding_end, pad_mode)}, {"Out"});
}
}
}
}
}
