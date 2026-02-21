// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"
namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// aten::l1_loss(Tensor input, Tensor target, int reduction=1) -> Tensor
OutputVector translate_l1_loss(const NodeContext& node) {
    auto a = node.get_input(0);
    auto b = node.get_input(1);
    align_eltwise_input_types(node, a, b);

    // reduction parse (accept any integral constant)
    auto reduction = std::string{"mean"};
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

    auto abs_diff = node.mark_node(std::make_shared<v0::Abs>(node.mark_node(std::make_shared<v1::Subtract>(a, b))));
    std::shared_ptr<ov::Node> out = abs_diff;
    if (reduction == "mean") {
        out = node.mark_node(std::make_shared<v1::ReduceMean>(out, get_axes_range(node, 0), false));
    } else if (reduction == "sum") {
        out = node.mark_node(std::make_shared<v1::ReduceSum>(out, get_axes_range(node, 0), false));
    }
    out = node.mark_node(std::make_shared<v1::ConvertLike>(out, a));
    return {out->output(0)};
}

OutputVector translate_l1_loss_fx(const NodeContext& node) {
    return translate_l1_loss(node);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
