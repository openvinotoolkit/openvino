// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_l2_loss_op(const NodeContext& node) {
    // L2Loss performs the following: output = sum(input ** 2) / 2
    default_op_checks(node, 1, {"L2Loss"});
    auto input = node.get_input(0);

    auto const_two = make_shared<Constant>(input.get_element_type(), Shape{}, 2);
    auto squared_input = make_shared<Power>(input, const_two);

    auto input_rank = compute_subgraph_scalar_rank(input, element::i32, true);
    auto const_zero = make_shared<Constant>(element::i32, Shape{}, 0);
    auto const_one = make_shared<Constant>(element::i32, Shape{}, 1);
    auto reduction_axes = make_shared<Range>(const_zero, input_rank, const_one, element::i32);
    auto squared_input_sum = make_shared<ReduceSum>(squared_input, reduction_axes);

    auto l2_loss = make_shared<Divide>(squared_input_sum, const_two);
    set_node_name(node.get_name(), l2_loss);
    return {l2_loss};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
