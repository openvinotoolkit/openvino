// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_select_base_op(const NodeContext& node,
                                      const Output<Node>& condition,
                                      const Output<Node>& x,
                                      const Output<Node>& y) {
    // at this point all inputs are NumPy broadcastable
    auto select = make_shared<v1::Select>(condition, x, y);
    set_node_name(node.get_name(), select);
    return {select};
}
bool has_complex_inputs(Output<Node>& x, Output<Node>& y, element::Type& complex_part_type) {
    auto complex_type_mark_x = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    auto complex_type_mark_y = as_type_ptr<ComplexTypeMark>(y.get_node_shared_ptr());
    if (complex_type_mark_x) {
        x = complex_type_mark_x->get_data();
        complex_part_type = complex_type_mark_x->get_complex_part_type();
    }
    if (complex_type_mark_y) {
        y = complex_type_mark_y->get_data();
        complex_part_type = complex_type_mark_y->get_complex_part_type();
    }
    return (complex_type_mark_x || complex_type_mark_y);
}
OutputVector translate_select_v2_op(const NodeContext& node) {
    // according to the TensorFlow documentation. See in the code:
    // https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/kernels/select.cc#L188-L211
    //
    // SelectV2 op selects values of 'x' if the corresponding value of 'condition'
    // is true or the value of 'y' if false. There are valid condition input sizes:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. Broadcastable shapes between 'condition', 'x' and 'y'.
    default_op_checks(node, 3, {"SelectV2", "SELECT_V2"}, true);
    auto condition = node.get_input(0);
    auto x = node.get_input(1);
    auto y = node.get_input(2);

    element::Type complex_part_type;
    auto is_complex = has_complex_inputs(x, y, complex_part_type);

    if (is_complex) {
        auto const_negative_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto new_condition = make_shared<v0::Unsqueeze>(condition, const_negative_one);
        auto result = translate_select_base_op(node, new_condition, x, y);
        auto complex_result = make_shared<ComplexTypeMark>(result[0].get_node_shared_ptr(), complex_part_type);
        return {complex_result->output(0)};
    } else {
        return translate_select_base_op(node, condition, x, y);
    }
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
    default_op_checks(node, 3, {"Select", "SELECT"}, true);
    auto condition = node.get_input(0);
    auto x = node.get_input(1);
    auto y = node.get_input(2);

    element::Type complex_part_type;
    auto is_complex = has_complex_inputs(x, y, complex_part_type);

    // compute number of dimensions to unsqueeze the condition
    auto cond_rank = compute_subgraph_scalar_rank(condition, element::i32);
    auto x_rank = compute_subgraph_scalar_rank(x, element::i32);
    auto num_new_axes = make_shared<v1::Subtract>(x_rank, cond_rank);

    // generate a new shape for the condition
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto new_subshape = make_shared<v3::Broadcast>(const_one, num_new_axes);
    auto cond_shape = make_shared<v3::ShapeOf>(condition, element::i32);
    // use extra dimensions in the begin to avoid concatenation of empty tensors that is not supported by Concat
    auto new_cond_shape = make_shared<v0::Concat>(OutputVector{const_one, cond_shape, new_subshape}, 0);

    // prepare the condition to have the same rank as operands `x` and `y`
    auto prep_cond = make_shared<v1::Reshape>(condition, new_cond_shape, false)->output(0);
    // squeeze prep_cond by one extra dimension specially added
    auto const_zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    prep_cond = make_shared<v0::Squeeze>(prep_cond, const_zero);

    auto result = translate_select_base_op(node, prep_cond, x, y);
    if (is_complex) {
        auto complex_result = make_shared<ComplexTypeMark>(result[0].get_node_shared_ptr(), complex_part_type);
        return {complex_result->output(0)};

    } else {
        return result;
    }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
