// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_ndtr(const NodeContext& context) {
    // aten::special_ndtr(Tensor self) -> Tensor
    // aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    // aten::ndtr(Tensor self) -> Tensor
    // aten::ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 2);
    auto x = get_input_with_floating_type(context, 0);

    // ndtr(x) = 0.5 * (1.0 + erf(x * (1.0 / sqrt(2.0))))
    const double inv_sqrt_2 = 0.70710678118654752440084436210485;
    auto const_alpha = context.mark_node(v0::Constant::create(element::f64, Shape{}, {inv_sqrt_2}));
    auto alpha = context.mark_node(std::make_shared<v1::ConvertLike>(const_alpha, x));
    auto x_scaled = context.mark_node(std::make_shared<v1::Multiply>(x, alpha));

    auto erf_node = context.mark_node(std::make_shared<v0::Erf>(x_scaled));

    auto const_one = context.mark_node(v0::Constant::create(element::f64, Shape{}, {1.0}));
    auto one = context.mark_node(std::make_shared<v1::ConvertLike>(const_one, x));
    auto add_one = context.mark_node(std::make_shared<v1::Add>(erf_node, one));

    auto const_half = context.mark_node(v0::Constant::create(element::f64, Shape{}, {0.5}));
    auto half = context.mark_node(std::make_shared<v1::ConvertLike>(const_half, x));
    Output<Node> res = context.mark_node(std::make_shared<v1::Multiply>(add_one, half));

    if (!context.input_is_none(1)) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, context.get_input(1)));
        context.mutate_input(1, res);
    }
    return {res};
}

}  // namespace ov::frontend::pytorch::op
