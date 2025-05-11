// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_l2_loss_op(const NodeContext& node) {
    // L2Loss performs the following: output = sum(input ** 2) / 2
    default_op_checks(node, 1, {"L2Loss"});
    auto input = node.get_input(0);

    auto const_two = create_same_type_const_scalar<float>(input, 2);
    auto squared_input = make_shared<v1::Power>(input, const_two);

    auto input_rank = compute_subgraph_scalar_rank(input, element::i32, true);
    auto const_zero = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto reduction_axes = make_shared<v4::Range>(const_zero, input_rank, const_one, element::i32);
    auto squared_input_sum = make_shared<v1::ReduceSum>(squared_input, reduction_axes);

    auto l2_loss = make_shared<v1::Divide>(squared_input_sum, const_two);
    set_node_name(node.get_name(), l2_loss);
    return {l2_loss};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
