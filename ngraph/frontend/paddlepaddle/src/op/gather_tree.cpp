// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <node_context.hpp>
#include "default_opset.hpp"
#include <iostream>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
using namespace opset8;
using namespace element;

NamedOutputs gather_tree(const NodeContext& node) {
    auto data_ids = node.get_ng_input("Ids");
    auto data_parents = node.get_ng_input("Parents");
    auto dtype = data_ids.get_element_type();
    PartialShape input_shape = data_ids.get_partial_shape();
    auto max_time = input_shape[0].get_length();

    // preparing max_seq_len
    const auto value_node_max_time = default_opset::Constant::create(dtype, {1}, {max_time});
    const auto ids_shape = std::make_shared<ShapeOf>(data_ids);
    const auto shape_node_max_seq_len = std::make_shared<StridedSlice>(ids_shape,
                                          Constant::create(i64, Shape{1}, std::vector<int64_t>{1}),
                                          Constant::create(i64, Shape{1}, std::vector<int64_t>{2}),
                                          std::vector<int64_t>{0},   // begin mask
                                          std::vector<int64_t>{0});  // end mask
    const auto max_seq_len = std::make_shared<default_opset::Broadcast>(value_node_max_time, shape_node_max_seq_len);

    // preparing end_token
    const auto value_node = default_opset::Constant::create(dtype, {1}, {0});
    const auto axis = ngraph::opset6::Constant::create(dtype, Shape{}, {0});
    auto end_token = std::make_shared<ngraph::opset6::Squeeze>(value_node, axis);

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::GatherTree>(data_ids, data_parents, max_seq_len, end_token)},
        {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph
