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
#include "flatten_contiguous_range.hpp"
#include <paddlepaddle_frontend/utility.hpp>
#include <ngraph/builder/reshape.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs flatten_contiguous_range (const NodeContext& node) {
    auto x_node = node.get_ng_input("X");
    auto shape_of_x = std::make_shared<opset6::ShapeOf>(x_node);
    int dims = x_node.get_partial_shape().rank().get_length();
    auto start_axis = node.get_attribute<int32_t>("start_axis");
    auto stop_axis = node.get_attribute<int32_t>("stop_axis");

    auto axis1_begin = opset6::Constant::create(element::i64, {1}, {0});
    auto axis1_end = opset6::Constant::create(element::i64, {1}, {start_axis});
    auto axis1 = std::make_shared<opset6::StridedSlice>(shape_of_x, axis1_begin, axis1_end, std::vector<int64_t>{0}, std::vector<int64_t>{0});
    OutputVector axes {axis1, opset6::Constant::create(element::i64, Shape{1}, {-1.0})};

    if (stop_axis < dims - 1) {
        auto axis2_begin = opset6::Constant::create(element::i64, {1}, {stop_axis + 1});
        auto axis2_end = opset6::Constant::create(element::i64, {1}, {dims});
        auto axis2_node = std::make_shared<opset6::StridedSlice>(shape_of_x, axis2_begin, axis2_end, std::vector<int64_t>{0}, std::vector<int64_t>{0});
        axes.push_back(axis2_node);
    }

    auto new_shape_node = std::make_shared<opset6::Concat>(axes, 0);
    return node.default_single_output_mapping({std::make_shared<opset6::Reshape>(x_node, new_shape_node, true)}, {"Out"});
}
}
}
}
}
