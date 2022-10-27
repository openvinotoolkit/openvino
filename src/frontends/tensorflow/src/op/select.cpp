// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset9;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_select_base_op(const NodeContext& node,
                                      const Output<Node>& condition,
                                      const Output<Node>& x,
                                      const Output<Node>& y) {
    // at this point all inputs are NumPy broadcastable
    auto select = make_shared<Select>(condition, x, y);
    set_node_name(node.get_name(), select);
    return {select};
}

OutputVector translate_select_v2_op(const NodeContext& node) {
    // according to the TensorFlow documentation. See in the code:
    // https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/kernels/select.cc#L188-L211
    //
    // SelectV2 op selects values of 'x' if the corresponding value of 'condition'
    // is true or the value of 'y' if false. There are valid condition input sizes:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. Broadcastable shapes between 'condition', 'x' and 'y'.
    default_op_checks(node, 3, {"SelectV2"});
    // no preparation for inputs are needed
    // inputs are already NumPy broadcastable
    return translate_select_base_op(node, node.get_input(0), node.get_input(1), node.get_input(2));
}

OutputVector translate_select_op(const NodeContext& node) {
    // according to the TensorFlow documentation. See in the code:
    // https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/kernels/select.cc#L188-L211
    //
    // Select op selects values of 'x' if the corresponding value of 'condition' is
    // true or the value of 'y' if false. There are valid condition input sizes:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. condition must be Rank 1 and match over the first dimension, or
    // 3. condition is scalar
    default_op_checks(node, 3, {"Select"});
    auto condition = node.get_input(0);
    auto x = node.get_input(1);
    auto y = node.get_input(2);
    auto cond_rank = condition.get_partial_shape().rank();
    TENSORFLOW_OP_VALIDATION(node,
                             cond_rank.is_static(),
                             "TensorFlow Frontend supports Select operation only with condition of static rank.");
    auto x_rank = x.get_partial_shape().rank();
    auto y_rank = y.get_partial_shape().rank();
    TENSORFLOW_OP_VALIDATION(node,
                             x_rank.is_static() || y_rank.is_static(),
                             "TensorFlow Frontend supports Select operation only with condition of static rank.");

    auto cond_rank_value = cond_rank.get_length();
    if (cond_rank_value == 0) {
        // the condition is a scalar that means all inputs are ready
        return translate_select_base_op(node, condition, x, y);
    }

    auto op_rank_value = (x_rank.is_static()) ? x_rank.get_length() : -1;
    if (y_rank.is_static() && op_rank_value > 0) {
        auto y_rank_value = y_rank.get_length();
        TENSORFLOW_OP_VALIDATION(
            node,
            op_rank_value == y_rank_value,
            "Internal TensorFlow Frontend error or incorrect model: x and y of Select must be of the same rank.");
    } else if (y_rank.is_static()) {
        op_rank_value = y_rank.get_length();
    }

    if (cond_rank_value == op_rank_value) {
        // there is nothing to unsqueeze
        // the condition is ready
        return translate_select_base_op(node, condition, x, y);
    }

    TENSORFLOW_OP_VALIDATION(node,
                             cond_rank_value <= op_rank_value,
                             "Internal TensorFlow Frontend error or incorrect model: rank of condition must be not "
                             "greater than ranks of x and y in Select.");

    // generate a range of indices [cond_rank_value, cond_rank_value + 1, ...]
    std::vector<int> axes(op_rank_value - cond_rank_value);
    std::iota(axes.begin(), axes.end(), cond_rank_value);
    auto unsqueeze_axes = make_shared<Constant>(element::i32, Shape{axes.size()}, axes);

    // prepare the condition by inserting the required dimensions
    auto prep_cond = make_shared<Unsqueeze>(condition, unsqueeze_axes);

    return translate_select_base_op(node, prep_cond, x, y);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
