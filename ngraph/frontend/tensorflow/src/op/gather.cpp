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

OutputVector TranslateGatherOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    auto ng_input_indices = node.get_ng_input(1);
    auto ng_axis = ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{}, 0);
    auto gather_op = ConstructNgNode<Gather>(node.get_name(), ng_input, ng_input_indices, ng_axis);
    return {gather_op};
}

OutputVector TranslateGatherV2Op(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    auto ng_input_coords = node.get_ng_input(1);
    auto ng_axis = node.get_ng_input(2);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto gather_op = ConstructNgNode<Gather>(node.get_name(), ng_input, ng_input_coords, ng_axis, batch_dims);
    return {gather_op};
}

OutputVector TranslateGatherNdOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto input_indices = node.get_ng_input(1);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto gathernd_op = make_shared<GatherND>(input, input_indices, batch_dims);
    gathernd_op->set_friendly_name(node.get_name());
    return gathernd_op->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph