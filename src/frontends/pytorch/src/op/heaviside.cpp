// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_heaviside(const NodeContext& context) {
    // aten::heaviside(Tensor input, Tensor values, *, Tensor? out=None) -> Tensor
    // heaviside(x, v) = 0 if x < 0; v if x == 0; 1 if x > 0
    // Torch evaluates this as `x == 0 ? v : (x > 0)`, so a NaN input yields 0.
    num_inputs_check(context, 2, 3);
    Output<Node> input;
    Output<Node> values;
    std::tie(input, values) = get_inputs_with_promoted_types(context, 0, 1);

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, input));
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, input));

    const auto is_zero = context.mark_node(std::make_shared<v1::Equal>(input, zero));
    const auto is_positive = context.mark_node(std::make_shared<v1::Greater>(input, zero));
    // x == x is false iff x is NaN. Needed because Greater(NaN, 0) is true here, while
    // torch's kernel `x == 0 ? v : (x > 0)` yields 0 for NaN.
    const auto is_self_equal = context.mark_node(std::make_shared<v1::Equal>(input, input));

    auto result = context.mark_node(std::make_shared<v1::Select>(is_positive, one, zero));
    result = context.mark_node(std::make_shared<v1::Select>(is_zero, values, result));
    result = context.mark_node(std::make_shared<v1::Select>(is_self_equal, result, zero));

    if (!context.input_is_none(2)) {
        context.mutate_input(2, result);
    }
    return {result};
}

}  // namespace ov::frontend::pytorch::op
