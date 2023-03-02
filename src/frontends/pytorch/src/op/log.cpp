// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_log(NodeContext& context) {
    // torch.log returns a tensor with the natural logarithm of the elements of input.
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
    auto log = context.mark_node(std::make_shared<v0::Log>(x));
    return {log};
};

OutputVector translate_log2(NodeContext& context) {
    // torch.log2 returns a tensor with the logarithm to the base 2 of the elements of input.
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto two = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2}));
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
    auto log2 = context.mark_node(std::make_shared<v0::Log>(two));
    auto log = context.mark_node(std::make_shared<v0::Log>(x));
    auto res = context.mark_node(std::make_shared<v1::Divide>(log, log2));
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov