// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector TranslateReduceOp(const NodeContext& node,
                               std::function<Output<Node>(Output<Node>, Output<Node>, const bool)> create_ng_node) {
    auto input = node.get_input(0);
    auto reduction_axes = node.get_input(1);
    auto tf_keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto res = create_ng_node(input, reduction_axes, tf_keep_dims);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}

template <typename T>
OutputVector translate_direct_reduce_op(const NodeContext& node) {
    // ensure its either an arithmetic or a logical reduction
    if (!(std::is_base_of<ov::op::util::ArithmeticReduction, T>::value ||
          std::is_base_of<ov::op::util::LogicalReduction, T>::value)) {
        TENSORFLOW_OP_VALIDATION(node,
                                 false,
                                 "Expected node to be either a valid logical or arithmetic reduction "
                                 "type");
    }
    return TranslateReduceOp(node, [](Output<Node> ng_input, Output<Node> ng_reduction_axes, const bool keep_dims) {
        return make_shared<T>(ng_input, ng_reduction_axes, keep_dims);
    });
}

template OutputVector translate_direct_reduce_op<ReduceLogicalOr>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceLogicalAnd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceMax>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceMean>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceMin>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceProd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceSum>(const NodeContext& node);
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov