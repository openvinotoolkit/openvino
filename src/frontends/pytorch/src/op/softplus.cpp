// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {

bool is_default_scalar(const NodeContext& context, size_t idx, double default_value) {
    if (context.input_is_none(idx)) {
        return true;
    }
    const auto constant = ov::util::get_constant_from_source(context.get_input(static_cast<int>(idx)));
    if (!constant || shape_size(constant->get_shape()) != 1) {
        return false;
    }
    return constant->cast_vector<double>()[0] == default_value;
}

}  // namespace

OutputVector translate_softplus(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    const bool default_beta = is_default_scalar(context, 1, 1.0);
    const bool default_threshold = is_default_scalar(context, 2, 20.0);
    if (default_beta && default_threshold) {
        return {context.mark_node(std::make_shared<v4::SoftPlus>(x))};
    }

    const auto input_type = x.get_element_type();
    const bool low_precision = input_type == element::f16 || input_type == element::bf16;
    Output<Node> input = x;
    if (low_precision) {
        input = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
    }

    Output<Node> scaled = input;
    Output<Node> beta;
    if (!default_beta) {
        beta = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(1), input));
        scaled = context.mark_node(std::make_shared<v1::Multiply>(input, beta));
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
    threshold = context.mark_node(std::make_shared<v1::ConvertLike>(threshold, input));
    auto above_threshold = context.mark_node(std::make_shared<v1::Greater>(scaled, threshold));
    Output<Node> result = context.mark_node(std::make_shared<v1::Select>(above_threshold, input, softplus));
    if (low_precision) {
        result = context.mark_node(std::make_shared<v0::Convert>(result, input_type));
    }
    return {result};
};

}  // namespace ov::frontend::pytorch::op
