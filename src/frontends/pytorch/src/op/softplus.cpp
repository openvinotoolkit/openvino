// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_softplus(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    const bool default_beta = context.input_is_none(1) || context.const_input<double>(1) == 1.0;
    const bool default_threshold = context.input_is_none(2) || context.const_input<double>(2) == 20.0;
    if (default_beta && default_threshold) {
        return {context.mark_node(std::make_shared<v4::SoftPlus>(x))};
    }

    Output<Node> scaled = x;
    Output<Node> beta;
    if (!default_beta) {
        beta = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(1), x));
        scaled = context.mark_node(std::make_shared<v1::Multiply>(x, beta));
    }
    Output<Node> softplus = context.mark_node(std::make_shared<v4::SoftPlus>(scaled));
    if (!default_beta) {
        softplus = context.mark_node(std::make_shared<v1::Divide>(softplus, beta));
    }

    Output<Node> threshold;
    if (default_threshold) {
        threshold = context.mark_node(v0::Constant::create(element::f32, Shape{}, {20.0f}));
    } else {
        threshold = context.get_input(2);
    }
    threshold = context.mark_node(std::make_shared<v1::ConvertLike>(threshold, x));
    auto above_threshold = context.mark_node(std::make_shared<v1::Greater>(scaled, threshold));
    return {context.mark_node(std::make_shared<v1::Select>(above_threshold, x, softplus))};
};

}  // namespace ov::frontend::pytorch::op
