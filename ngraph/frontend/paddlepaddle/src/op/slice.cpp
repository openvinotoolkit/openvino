// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits.h>

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs slice(const NodeContext& node) {
    auto data = node.get_ng_input("Input");
    auto axes = node.get_attribute<std::vector<int32_t>>("axes");
    Output<Node> start_idx_node, end_idx_node;
    if (node.has_ng_input("StartsTensor")) {
        start_idx_node = node.get_ng_input("StartsTensor");
    } else if (node.has_ng_input("StartsTensorList")) {
        auto inputs = node.get_ng_inputs("StartsTensorList");
        start_idx_node = std::make_shared<ngraph::opset6::Concat>(inputs, 0);
    } else {
        auto starts = node.get_attribute<std::vector<int32_t>>("starts");
        start_idx_node = opset6::Constant::create(element::i32, {starts.size()}, starts);
    }

    if (node.has_ng_input("EndsTensor")) {
        end_idx_node = node.get_ng_input("EndsTensor");
    } else if (node.has_ng_input("EndsTensorList")) {
        auto inputs = node.get_ng_inputs("EndsTensorList");
        end_idx_node = std::make_shared<ngraph::opset6::Concat>(inputs, 0);
    } else {
        auto ends = node.get_attribute<std::vector<int32_t>>("ends");
        end_idx_node = opset6::Constant::create(element::i32, {ends.size()}, ends);
    }

    // The following process is:
    // Given:
    // data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] // shape is: [2, 4]
    // axes = [0]
    // starts = [1]
    // ends = [2]
    // Our process is:
    //  1. Get 'axes': [0, 1], 'starts', 'ends'
    //  2. Get data shape: [2,4] and dims: 2
    //  3. Create two tensor t1 and t2, shape is the dims from step2: 2. t1: [0, 0], t2: [INT_MAX, INT_MAX]
    //  4. Use 'ScatterNDUpdate' to update some elements in t1, the updated indexes are coming from 'axes', the contents
    //  are coming from 'starts', t1: [1, 0]; apply the similar process to t2
    //  5. Call 'StrideSlice' with t1 and t2
    // Why using ScatterNDUpdate is that 'axes' may be discontinuous.

    // the shape of input, such as [2, 4]
    auto shape_node = std::make_shared<opset6::ShapeOf>(data, element::Type_t::i32);
    // the input dim, such as [2]
    auto shape_shape_node = std::make_shared<opset6::ShapeOf>(shape_node, element::i32);
    auto const_0_node = opset6::Constant::create(element::i32, {}, {0});
    auto const_max_node = opset6::Constant::create(element::i32, {}, {INT_MAX});
    // t1: [0, 0]
    auto start_node = std::make_shared<opset6::Broadcast>(const_0_node, shape_shape_node);
    // t2: [INT_MAX, INT_MAX]
    auto end_node = std::make_shared<opset6::Broadcast>(const_max_node, shape_shape_node);
    auto axes_node = opset6::Constant::create(element::i32, {axes.size(), 1}, axes);
    // update t1
    auto fixed_start_node = std::make_shared<opset6::ScatterNDUpdate>(start_node, axes_node, start_idx_node);
    // update t2
    auto fixed_end_node = std::make_shared<opset6::ScatterNDUpdate>(end_node, axes_node, end_idx_node);

    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::StridedSlice>(data,
                                                                                              fixed_start_node,
                                                                                              fixed_end_node,
                                                                                              std::vector<int64_t>{0},
                                                                                              std::vector<int64_t>{0})},
                                              {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph