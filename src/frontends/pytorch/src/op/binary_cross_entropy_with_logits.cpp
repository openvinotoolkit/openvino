// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;

OutputVector translate_binary_cross_entropy_with_logits(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto target = context.get_input(1);

    auto zero = v0::Constant::create(input.get_element_type(), Shape{}, {0.0});
    auto one = v0::Constant::create(input.get_element_type(), Shape{}, {1.0});
    auto neg_one = v0::Constant::create(input.get_element_type(), Shape{}, {-1.0});

    auto max_val = context.mark_node(std::make_shared<v1::Maximum>(input, zero));

    auto abs_input = context.mark_node(std::make_shared<v0::Abs>(input));
    auto neg_abs_input = context.mark_node(std::make_shared<v1::Multiply>(abs_input, neg_one));

    auto exp_neg_abs = context.mark_node(std::make_shared<v0::Exp>(neg_abs_input));
    auto exp_plus_one = context.mark_node(std::make_shared<v1::Add>(exp_neg_abs, one));
    auto log_term = context.mark_node(std::make_shared<v0::Log>(exp_plus_one));

    auto input_mul_target = context.mark_node(std::make_shared<v1::Multiply>(input, target));

    auto temp_sub = context.mark_node(std::make_shared<v1::Subtract>(max_val, input_mul_target));
    auto loss = context.mark_node(std::make_shared<v1::Add>(temp_sub, log_term));

    return {loss};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

