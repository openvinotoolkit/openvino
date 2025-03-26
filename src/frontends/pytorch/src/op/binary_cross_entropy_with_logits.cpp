// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
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

    auto sigmoid = context.mark_node(std::make_shared<ov::op::v0::Sigmoid>(input));
    auto log_sigmoid = context.mark_node(std::make_shared<ov::op::v0::Log>(sigmoid));

    auto one = ov::op::v0::Constant::create(target.get_element_type(), Shape{}, {1.0});
    auto one_minus_target = context.mark_node(std::make_shared<ov::op::v1::Subtract>(one, target));
    auto one_minus_sigmoid = context.mark_node(std::make_shared<ov::op::v1::Subtract>(one, sigmoid));
    auto log_one_minus_sigmoid = context.mark_node(std::make_shared<ov::op::v0::Log>(one_minus_sigmoid));

    // loss = - (target * log_sigmoid + (1 - target) * log(1 - sigmoid))
    auto positive_term = context.mark_node(std::make_shared<ov::op::v1::Multiply>(target, log_sigmoid));
    auto negative_term = context.mark_node(std::make_shared<ov::op::v1::Multiply>(one_minus_target, log_one_minus_sigmoid));
    auto sum = context.mark_node(std::make_shared<ov::op::v1::Add>(positive_term, negative_term));

    auto neg_one = ov::op::v0::Constant::create(sum->get_element_type(), Shape{}, {-1.0});
    auto loss = context.mark_node(std::make_shared<ov::op::v1::Multiply>(sum, neg_one));

    return {loss};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
