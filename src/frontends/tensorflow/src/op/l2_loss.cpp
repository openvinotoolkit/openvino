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

OutputVector translate_l2_loss_op(const NodeContext& node) {
    auto input = node.get_input(0);

    vector<float> val;
    val.push_back(2.0);
    auto const_2 = make_shared<Constant>(input.get_element_type(), Shape{}, 2);
    auto pow = make_shared<Multiply>(input, input);
    auto p_shape = input.get_partial_shape();

    Output<Node> reduction_axes;
    if (p_shape.rank().is_static()) {
        vector<int64_t> axes(p_shape.size());
        iota(axes.begin(), axes.end(), 0);
        reduction_axes = make_shared<Constant>(element::i64, Shape{axes.size()}, axes);
    } else {
        auto shape = make_shared<ShapeOf>(input);
        auto rank = make_shared<ShapeOf>(shape);
        auto start = make_shared<Constant>(element::i64, Shape{1}, 0);
        auto step = make_shared<Constant>(element::i64, Shape{1}, 1);
        reduction_axes = make_shared<Range>(start, rank, step, element::i64);
    }

    auto sum = make_shared<ReduceSum>(pow, reduction_axes);
    auto res = make_shared<Divide>(sum, const_2);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
