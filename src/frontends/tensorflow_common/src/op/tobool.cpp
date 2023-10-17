// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_tobool_op(const NodeContext& node) {
    // (rank(x) == 0 && x != 0) || (rank > 0 && ReduceProd(ShapeOf(x))) > 0

    default_op_checks(node, 2, {"ToBool"});
    auto x = node.get_input(0);

    // prepare auxiliary zero and one constants of the same type as the inputs
    auto zero = create_same_type_const_scalar<int32_t>(x, 0);
    auto true_const = create_same_type_const_scalar<int32_t>(x, true);
    auto false_const = create_same_type_const_scalar<int32_t>(x, false);

    // compute a mask to get rank(x) == 0
    auto x_rank = compute_subgraph_scalar_rank(x, element::i32);

    // compute rank(x) == 0
    auto is_zero = make_shared<v1::Equal>(x_rank, zero);

    // compute mask to get x != 0
    auto is_not_zero = make_shared<v1::NotEqual>(x, zero);

    // compute (rank(x) == 0 && x != 0)
    auto logical_and = make_shared<v1::LogicalAnd>(is_zero, is_not_zero);
    
    // compute rank(x) > 0
    auto greater_than_zero = make_shared<v1::Greater>(x_rank, zero);

    // compute ShapeOf(x)
    auto cond_shape = make_shared<v3::ShapeOf>(x, element::i32);
    
    // compute ReduceProd(ShapeOf(x)))
    auto reduce_prod = make_shared<v1::ReduceProd>(cond_shape);

    // compute ReduceProd(ShapeOf(x))) > 0
    auto greater_than__zero_2 = make_shared<v1::Greater>(reduce_prod, zero);
    
    // compute (rank > 0 && ReduceProd(ShapeOf(x))) > 0
    auto logical_and_2 = make_shared<v1::LogicalAnd>(greater_than_zero, greater_than__zero_2);

    auto logical_or = make_shared<v1::LogicalOr>(logical_and, logical_and_2);

    auto tobool = make_shared<v1::Select>(logical_or, true_const, false_const);
    set_node_name(node.get_name(), tobool);
    return tobool->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov