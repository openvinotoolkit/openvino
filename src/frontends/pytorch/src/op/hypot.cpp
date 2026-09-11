// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_hypot(const NodeContext& context) {
    // hypot(input, other, *) -> Tensor : elementwise sqrt(input^2 + other^2), with broadcasting.
    num_inputs_check(context, 2, 2);
    Output<Node> lhs;
    Output<Node> rhs;
    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);

    const auto lhs_squared = context.mark_node(std::make_shared<v1::Multiply>(lhs, lhs));
    const auto rhs_squared = context.mark_node(std::make_shared<v1::Multiply>(rhs, rhs));
    const auto sum_of_squares = context.mark_node(std::make_shared<v1::Add>(lhs_squared, rhs_squared));
    const auto result = context.mark_node(std::make_shared<v0::Sqrt>(sum_of_squares));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
