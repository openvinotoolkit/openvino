// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_norm(NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto input_tensor = context.get_input(0);
    auto p = context.const_input<float>(1);
    auto dim = context.get_input(2);
    auto keep_dim = context.const_input<bool>(3);

    Output<Node> res;
    if (p == 1) {
        res = context.mark_node(std::make_shared<v4::ReduceL1>(input_tensor, dim, keep_dim));
    } else if (p == 2) {
        res = context.mark_node(std::make_shared<v4::ReduceL2>(input_tensor, dim, keep_dim));
    } else if (p == std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(abs, dim, keep_dim));
    } else if (p == -std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(abs, dim, keep_dim));
    } else {
        auto const_p = context.mark_node(v0::Constant::create(element::f64, Shape{1}, {p}));
        auto const_p_inv = context.mark_node(v0::Constant::create(element::f64, Shape{1}, {1.0 / p}));
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto pow = context.mark_node(std::make_shared<v1::Power>(abs, const_p));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(pow, dim, keep_dim));
        res = context.mark_node(std::make_shared<v1::Power>(sum, const_p_inv));
    }
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov