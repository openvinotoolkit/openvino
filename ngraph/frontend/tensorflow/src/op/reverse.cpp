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

OutputVector TranslateReverseOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto axes = node.get_ng_input(1);

    auto axes_const = dynamic_pointer_cast<Constant>(axes.get_node_shared_ptr());
    TF_OP_VALIDATION_CHECK(node, axes_const != nullptr, "Axes input must be constant.");
    TF_OP_VALIDATION_CHECK(node, axes_const->get_shape().size() == 1, "Axes input must be 1D.");
    TF_OP_VALIDATION_CHECK(node, axes_const->get_shape()[0] == 1, "Axes input must have only one value.");
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

    auto reverse = make_shared<ReverseSequence>(input, seq_lengths, batch_axis, seq_axis);
    reverse->set_friendly_name(node.get_name());
    return reverse->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
