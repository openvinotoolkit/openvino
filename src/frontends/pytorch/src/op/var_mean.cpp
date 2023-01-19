// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_var_mean(NodeContext& context) {
    auto data = context.get_input(0);
    bool unbiased = true;
    bool keepdims = false;
    auto num_elements = numel(context, 0);
    bool keepdim_mean;
    std::shared_ptr<ov::Node> mean, t_mean;
    ov::Output<ov::Node> axes;
    if (context.inputs().size() == 2) {
        // aten::var_mean(input, unbiased)
        axes = context.mark_node(get_axes_range(context, 0));
        unbiased = context.const_input<bool>(1);
        mean = context.mark_node(std::make_shared<opset10::ReduceMean>(data, axes, keepdims));
        t_mean = mean;
        keepdim_mean = keepdims;
    } else {
        // aten::var_mean(input, dim, unbiased:bool=None, keepdim:bool=None)
        if (!context.input_is_none(2)) {
            unbiased = context.const_input<bool>(2);
        }
        if (!context.input_is_none(3)) {
            keepdims = context.const_input<bool>(3);
        }
        if (context.input_is_none(1)) {
            axes = context.mark_node(get_axes_range(context, 0));
            mean = context.mark_node(std::make_shared<opset10::ReduceMean>(data, axes, keepdims));
            t_mean = mean;
        } else {
            axes = context.get_input(1);
            mean = context.mark_node(std::make_shared<opset10::ReduceMean>(data, axes, keepdims));
            t_mean = context.mark_node(std::make_shared<opset10::ReduceMean>(data, axes, true));
            auto reduced_dims = context.mark_node(std::make_shared<opset10::ShapeOf>(data));
            auto zero = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {0}));
            reduced_dims = context.mark_node(std::make_shared<opset10::Gather>(reduced_dims, axes, zero));
            num_elements = context.mark_node(std::make_shared<opset10::ReduceProd>(reduced_dims, zero, false));
        }
        keepdim_mean = context.input_is_none(1) ? false : keepdims;
    }
    auto sub_v = context.mark_node(std::make_shared<opset10::Subtract>(data, t_mean));
    auto sqr_sub = context.mark_node(std::make_shared<opset10::Multiply>(sub_v, sub_v));
    auto var = context.mark_node(std::make_shared<opset10::ReduceMean>(sqr_sub, axes, keepdim_mean));
    // if unbiased=true Bessel’s correction will be used
    // Correct bias in calculating variance, by dividing it over (N - 1) instead on N
    if (unbiased) {
        num_elements = context.mark_node(std::make_shared<opset10::ConvertLike>(num_elements, data));
        auto one = context.mark_node(opset10::Constant::create(element::f32, Shape{}, {1}));
        one = context.mark_node(std::make_shared<opset10::ConvertLike>(one, data));
        auto mul = context.mark_node(std::make_shared<opset10::Multiply>(var, num_elements));
        auto n_minus_one = context.mark_node(std::make_shared<opset10::Subtract>(num_elements, one));
        var = context.mark_node(std::make_shared<opset10::Divide>(mul, n_minus_one));
    }
    return {var, mean};
};

OutputVector translate_var(NodeContext& context) {
    auto res = translate_var_mean(context);
    return {res[0]};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov