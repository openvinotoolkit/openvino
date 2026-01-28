// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cmath>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// aten::smooth_l1_loss(Tensor input, Tensor target, int reduction=1, float beta=1.0) -> Tensor
OutputVector translate_smooth_l1_loss(const NodeContext& node) {
    // Inputs
    auto input = node.get_input(0);
    auto target = node.get_input(1);

    // Align dtypes
    target = node.mark_node(std::make_shared<v1::ConvertLike>(target, input));
    align_eltwise_input_types(node, input, target);
    auto a = input;
    auto b = target;

    // Scalar constants (float32; aligned via ConvertLike)
    auto c05 = v0::Constant::create(element::f32, Shape{}, {0.5f});
    auto c05_like = node.mark_node(std::make_shared<v1::ConvertLike>(c05, a));
    auto eps = v0::Constant::create(element::f32, Shape{}, {1e-5f});
    auto eps_like = node.mark_node(std::make_shared<v1::ConvertLike>(eps, a));

    // Beta parameter: input[3] (FX) or attribute (TS)
    ov::Output<ov::Node> beta_like;
    if (node.get_input_size() > 3 && !node.input_is_none(3)) {
        beta_like = node.mark_node(std::make_shared<v1::ConvertLike>(node.get_input(3), a));
    } else {
        float beta_attr = node.get_attribute<float>("beta", 1.0f);
        OPENVINO_ASSERT(beta_attr >= 0.f, "smooth_l1_loss: beta must be non-negative");
        beta_like = node.mark_node(
            std::make_shared<v1::ConvertLike>(v0::Constant::create(element::f32, Shape{}, {beta_attr}), a));
    }

    // Per-element loss
    auto diff = node.mark_node(std::make_shared<v1::Subtract>(a, b));
    auto l1 = node.mark_node(std::make_shared<v0::Abs>(diff));
    auto is_small =
        node.mark_node(std::make_shared<v1::Less>(node.mark_node(std::make_shared<v0::Abs>(beta_like)), eps_like));

    // Quadratic region: 0.5 * diff^2 / beta
    auto diff_sq = node.mark_node(std::make_shared<v1::Multiply>(diff, diff));
    auto quad_num = node.mark_node(std::make_shared<v1::Multiply>(c05_like, diff_sq));
    auto quad = node.mark_node(std::make_shared<v1::Divide>(quad_num, beta_like));
    // Linear region: |x| - 0.5 * beta
    auto lin = node.mark_node(
        std::make_shared<v1::Subtract>(l1, node.mark_node(std::make_shared<v1::Multiply>(c05_like, beta_like))));

    auto elem = node.mark_node(
        std::make_shared<v1::Select>(node.mark_node(std::make_shared<v1::Less>(l1, beta_like)), quad, lin));
    auto safe = node.mark_node(std::make_shared<v1::Select>(is_small, l1, elem));

    // Reduction (0/1/2 -> none/mean/sum) or attribute
    std::string reduction = "mean";
    if (node.get_input_size() > 2 && !node.input_is_none(2)) {
        auto red_input = node.get_input(2);
        if (auto red_const = std::dynamic_pointer_cast<v0::Constant>(red_input.get_node_shared_ptr())) {
            int64_t red_val = 1;
            if (red_const->get_element_type().is_integral_number()) {
                red_val = red_const->cast_vector<int64_t>()[0];
            }
            reduction = (red_val == 0) ? "none" : (red_val == 1) ? "mean" : (red_val == 2) ? "sum" : reduction;
        }
    } else {
        reduction = node.get_attribute<std::string>("reduction", "mean");
    }

    if (reduction == "none") {
        auto out = node.mark_node(std::make_shared<v1::ConvertLike>(safe, a));
        return {out->output(0)};
    }

    auto axes = get_axes_range(node, 0);
    if (reduction == "mean") {
        auto out = node.mark_node(std::make_shared<v1::ReduceMean>(safe, axes, false));
        out = node.mark_node(std::make_shared<v1::ConvertLike>(out, a));
        return {out->output(0)};
    }
    auto out = node.mark_node(std::make_shared<v1::ReduceSum>(safe, axes, false));
    out = node.mark_node(std::make_shared<v1::ConvertLike>(out, a));
    return {out->output(0)};
}

OutputVector translate_smooth_l1_loss_fx(const NodeContext& node) {
    return translate_smooth_l1_loss(node);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov