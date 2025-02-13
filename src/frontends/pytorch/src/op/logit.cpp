// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/clamp.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_logit(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    if (!context.input_is_none(1)) {
        auto eps = context.const_input<double>(1);
        const auto one_minus_eps = (double)1 - eps;
        x = context.mark_node(std::make_shared<v0::Clamp>(x, eps, one_minus_eps));
    }

    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, x));
    auto one_minus_x = context.mark_node(std::make_shared<v1::Subtract>(one, x));
    auto division = context.mark_node(std::make_shared<v1::Divide>(x, one_minus_x));
    return {context.mark_node(std::make_shared<v0::Log>(division))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov