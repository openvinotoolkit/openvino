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

OutputVector translate_reverse_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto axes = node.get_input(1);

    auto axes_const = dynamic_pointer_cast<Constant>(axes.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(node, axes_const != nullptr, "Axes input must be constant.");
    TENSORFLOW_OP_VALIDATION(node, axes_const->get_shape().size() == 1, "Axes input must be 1D.");
    TENSORFLOW_OP_VALIDATION(node, axes_const->get_shape()[0] == 1, "Axes input must have only one value.");
    auto seq_axis = axes_const->cast_vector<int64_t>().at(0);
    int64_t batch_axis = !seq_axis;

    Output<Node> seq_lengths;
    if (input.get_partial_shape().is_static()) {
        auto in_shape = input.get_shape();
        seq_lengths = make_shared<Constant>(element::i64, Shape{in_shape[batch_axis]}, in_shape[seq_axis]);
    } else {
        auto shape = make_shared<ShapeOf>(input);
        auto one = make_shared<Constant>(element::i64, Shape{1}, 1);
        auto gather_batch = make_shared<Gather>(shape,
                                                make_shared<Constant>(element::i64, Shape{1}, batch_axis),
                                                make_shared<Constant>(element::i64, Shape{1}, 0));
        auto gather_seq = make_shared<Gather>(shape,
                                              make_shared<Constant>(element::i64, Shape{1}, seq_axis),
                                              make_shared<Constant>(element::i64, Shape{1}, 0));
        auto broadcast = make_shared<Broadcast>(one, gather_batch);
        seq_lengths = make_shared<Multiply>(broadcast, gather_seq);
    }

    auto res = make_shared<ReverseSequence>(input, seq_lengths, batch_axis, seq_axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
