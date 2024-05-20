// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"

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

OutputVector translate_select_v2_op(const NodeContext& node) {
    // according to the TensorFlow documentation. See in the code:
    // https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/kernels/select.cc#L188-L211
    //
    // SelectV2 op selects values of 'x' if the corresponding value of 'condition'
    // is true or the value of 'y' if false. There are valid condition input sizes:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. Broadcastable shapes between 'condition', 'x' and 'y'.
    default_op_checks(node, 3, {"SelectV2", "SELECT_V2"});
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
    default_op_checks(node, 3, {"Select", "SELECT"});
    auto select = make_shared<v1::Select>(node.get_input(0),
                                          node.get_input(1),
                                          node.get_input(2),
                                          AutoBroadcastSpec(AutoBroadcastType::PDPD, 0));
    set_node_name(node.get_name(), select);
    return {select};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
