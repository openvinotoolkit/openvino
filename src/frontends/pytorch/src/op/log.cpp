// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

std::shared_ptr<ov::Node> translate_log_sigmoid_common(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto op_vector = op::translate_1to1_match_1_inputs_with_fp32_type_alignment<v0::Sigmoid>(context);
    PYTORCH_OP_CONVERSION_CHECK(op_vector.size() == 1,
                                "Expected exactly one element in the vector. Got: ",
                                op_vector.size());
    auto sigmoid = op_vector[0];
    auto log = context.mark_node(std::make_shared<v0::Log>(sigmoid));
    return log;
};

OutputVector translate_log_sigmoid(const NodeContext& context) {
    return {translate_log_sigmoid_common(context)};
};

OutputVector translate_log_sigmoid_fx(const NodeContext& context) {
    auto log = translate_log_sigmoid_common(context);
    return {context.mark_node(make_list_construct(log->outputs()))};
};

OutputVector translate_log2(const NodeContext& context) {
    // torch.log2 returns a tensor with the logarithm to the base 2 of the elements of input.
    num_inputs_check(context, 1, 2);
    auto op_vector = op::translate_1to1_match_1_inputs_with_fp32_type_alignment<v0::Log>(context);
    PYTORCH_OP_CONVERSION_CHECK(op_vector.size() == 1,
                                "Expected exactly one element in the vector. Got: ",
                                op_vector.size());
    auto log = op_vector[0];

    auto two = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2}));
    two = context.mark_node(std::make_shared<v1::ConvertLike>(two, log));
    auto log2 = context.mark_node(std::make_shared<v0::Log>(two));

    auto res = context.mark_node(std::make_shared<v1::Divide>(log, log2));
    return {res};
};

OutputVector translate_log10(const NodeContext& context) {
    // torch.log10 returns a tensor with the logarithm to the base 10 of the elements of input.
    num_inputs_check(context, 1, 2);
    auto op_vector = op::translate_1to1_match_1_inputs_with_fp32_type_alignment<v0::Log>(context);
    PYTORCH_OP_CONVERSION_CHECK(op_vector.size() == 1,
                                "Expected exactly one element in the vector. Got: ",
                                op_vector.size());
    auto log = op_vector[0];

    auto ten = context.mark_node(v0::Constant::create(element::f32, Shape{}, {10}));
    ten = context.mark_node(std::make_shared<v1::ConvertLike>(ten, log));
    auto log10 = context.mark_node(std::make_shared<v0::Log>(ten));

    auto res = context.mark_node(std::make_shared<v1::Divide>(log, log10));
    return {res};
};

OutputVector translate_logsumexp(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    ov::Output<ov::Node> dim;
    if (!context.input_is_none(1)) {
        dim = context.get_input(1);
    } else {
        dim = context.mark_node(get_axes_range(context, 0));
    }
    bool keepdim = false;
    if (!context.input_is_none(2)) {
        keepdim = context.const_input<bool>(2);
    }
    // for numerical stability and avoiding exponent explosion,
    // apply the following formula:
    // ln (e^x1 + ... +e^xn) = ln (e^k (e^(x1-k) + ... +e^(xn-k))) =
    // = k + ln (e^(x1-k) + ... +e^(xn-k)), where k = max (x1, ..., xn)
    // by this trick, we reach exponent degree <= 0
    auto k = context.mark_node(std::make_shared<v1::ReduceMax>(input, dim, true));
    auto input_minus_k = context.mark_node(std::make_shared<v1::Subtract>(input, k));
    auto exp = context.mark_node(std::make_shared<v0::Exp>(input_minus_k));
    auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(exp, dim, keepdim));
    auto log = context.mark_node(std::make_shared<v0::Log>(sum));
    if (!keepdim) {
        k = context.mark_node(std::make_shared<v0::Squeeze>(k, dim));
    }
    auto logsumexp = context.mark_node(std::make_shared<v1::Add>(k, log));
    return {logsumexp};
};

OutputVector translate_log1p(const NodeContext& context) {
    // torch.log1p returns a tensor with the natural logarithm of the elements of input + 1.
    num_inputs_check(context, 1, 2);
    auto x = get_input_with_floating_type(context, 0);
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}))->output(0);
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, x));
    auto x_plus_one = context.mark_node(std::make_shared<v1::Add>(x, one));
    auto log = context.mark_node(std::make_shared<v0::Log>(x_plus_one));
    return {log};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
