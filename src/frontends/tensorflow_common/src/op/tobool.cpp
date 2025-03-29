// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_tobool_op(const NodeContext& node) {
    // (rank(x) == 0 && x != 0) || (rank > 0 && ReduceProd(ShapeOf(x))) > 0
    default_op_checks(node, 1, {"ToBool"});
    auto x = node.get_input(0);

    // prepare auxiliary zero and zero constants of the same type as the inputs
    auto zero_x = create_same_type_const_scalar<int32_t>(x, 0);
    auto zero_i64 = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto one_i64 = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto false_const = make_shared<v0::Constant>(element::boolean, Shape{}, false);
    // compute a mask to get rank(x) == 0
    auto x_rank = compute_subgraph_scalar_rank(x, element::i64, true);

    // 1. try to evaluate if it satisfy non-zero scalar input
    // compute rank(x) == 0
    auto is_rank_zero = make_shared<v1::Equal>(x_rank, zero_i64);
    // compute mask to get x != 0
    auto is_x_not_zero = make_shared<v1::NotEqual>(x, zero_x)->output(0);
    // compute (rank(x) == 0 && x != 0)
    auto scalar_cond = make_shared<v1::LogicalAnd>(is_rank_zero, is_x_not_zero)->output(0);
    // generate reduce_axes
    auto reduce_axes = make_shared<v0::Range>(zero_i64, x_rank, one_i64);
    scalar_cond = make_shared<v1::ReduceLogicalAnd>(scalar_cond, reduce_axes, false);
    // correct result for empty tensor, for which scalar_cond is still equal to True
    scalar_cond = make_shared<v1::Select>(is_rank_zero, scalar_cond, false_const);

    // 2. try to evaluate if it is non-scalar input tensor and not empty tensor
    // compute rank(x) > 0
    auto rank_greater_than_zero = make_shared<v1::Greater>(x_rank, zero_i64);
    // compute ShapeOf(x)
    auto x_shape = make_shared<v3::ShapeOf>(x, element::i64);
    // compute ReduceProd(ShapeOf(x))) and axis
    auto reduce_axis = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto num_elems = make_shared<v1::ReduceProd>(x_shape, reduce_axis, false);
    // compute ReduceProd(ShapeOf(x))) > 0
    auto num_elems_greater_than_zero = make_shared<v1::Greater>(num_elems, zero_i64);
    // compute (rank > 0 && ReduceProd(ShapeOf(x))) > 0
    // it will be a scalar
    auto non_scalar_tensor_not_empty = make_shared<v1::LogicalAnd>(rank_greater_than_zero, num_elems_greater_than_zero);

    auto to_bool = make_shared<v1::LogicalOr>(scalar_cond, non_scalar_tensor_not_empty);
    set_node_name(node.get_name(), to_bool);
    return to_bool->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
