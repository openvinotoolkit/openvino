// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateSplitOp(const NodeContext& node) {
    auto axes = node.get_ng_input(0);
    auto input = node.get_ng_input(1);
    auto num_split = node.get_attribute<int64_t>("num_split");

    auto ng_split = make_shared<Split>(input, axes, num_split);
    ng_split->set_friendly_name(node.get_name());
    return ng_split->outputs();
}

OutputVector TranslateSplitVOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto split_lengths = node.get_ng_input(1);
    auto split_dims = node.get_ng_input(2);

    // todo(itikhono): double check split_lengths and split_dims are in supported form here
    auto split_v = make_shared<VariadicSplit>(input, split_dims, split_lengths);
    split_v->set_friendly_name(node.get_name());
    return split_v->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
