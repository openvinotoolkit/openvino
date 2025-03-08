// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector translate_var_mean_common(const NodeContext& context,
                                       const Output<Node>& data,
                                       const Output<Node>& axes,
                                       int32_t correction,
                                       bool keepdims) {
    auto num_elements = numel(context, data);
    std::shared_ptr<ov::Node> mean, t_mean;
    auto _axes = axes;
    if (!_axes.get_node()) {
        // aten::var_mean(input, unbiased)
        _axes = context.mark_node(get_axes_range(context, 0));
        mean = context.mark_node(std::make_shared<v1::ReduceMean>(data, _axes, keepdims));
        t_mean = mean;
    } else {
        mean = context.mark_node(std::make_shared<v1::ReduceMean>(data, _axes, keepdims));
        t_mean = context.mark_node(std::make_shared<v1::ReduceMean>(data, _axes, true));
        auto reduced_dims = context.mark_node(std::make_shared<v3::ShapeOf>(data, element::i32));
        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        reduced_dims = context.mark_node(std::make_shared<v8::Gather>(reduced_dims, _axes, zero));
        num_elements = context.mark_node(std::make_shared<v1::ReduceProd>(reduced_dims, zero, false));
    }
    auto sub_v = context.mark_node(std::make_shared<v1::Subtract>(data, t_mean));
    auto sqr_sub = context.mark_node(std::make_shared<v1::Multiply>(sub_v, sub_v));
    auto var = context.mark_node(std::make_shared<v1::ReduceMean>(sqr_sub, _axes, keepdims));
    // if unbiased=true Bessel’s correction will be used
    // Correct bias in calculating variance, by dividing it over (N - 1) instead on N
    if (correction) {
        PYTORCH_OP_CONVERSION_CHECK(correction == 1, "Unexpected value of correction.");
        num_elements = context.mark_node(std::make_shared<v1::ConvertLike>(num_elements, data));
        auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
        one = context.mark_node(std::make_shared<v1::ConvertLike>(one, data));
        auto mul = context.mark_node(std::make_shared<v1::Multiply>(var, num_elements));
        auto n_minus_one = context.mark_node(std::make_shared<v1::Subtract>(num_elements, one));
        var = context.mark_node(std::make_shared<v1::Divide>(mul, n_minus_one));
    }
    return {var, mean};
};
}  // namespace

OutputVector translate_var_mean(const NodeContext& context) {
    num_inputs_check(context, 1, 4);
    auto data = context.get_input(0);
    bool unbiased = true;
    bool keepdim = false;
    ov::Output<ov::Node> axes;
    if (context.inputs().size() == 2) {
        // aten::var_mean(input, unbiased)
        axes = context.mark_node(get_axes_range(context, 0));
        unbiased = context.const_input<bool>(1);
    } else {
        // aten::var_mean(input, dim, unbiased:bool=None, keepdim:bool=None)
        if (!context.input_is_none(2)) {
            unbiased = context.const_input<bool>(2);
        }
        if (!context.input_is_none(3)) {
            keepdim = context.const_input<bool>(3);
        }
        if (!context.input_is_none(1)) {
            axes = context.get_input(1);
        }
    }
    auto res = translate_var_mean_common(context, data, axes, static_cast<int32_t>(unbiased), keepdim);
    return res;
};

OutputVector translate_var_mean_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto data = context.get_input(0);
    ov::Output<ov::Node> axes;

    if (!context.input_is_none(1)) {
        axes = context.get_input(1);
    }
    int32_t correction = 0;
    if (context.has_attribute("correction")) {
        correction = context.get_attribute<int32_t>("correction");
    }
    bool keepdim = false;
    if (context.has_attribute("keepdim")) {
        keepdim = context.get_attribute<bool>("keepdim");
    }
    auto res = translate_var_mean_common(context, data, axes, correction, keepdim);
    return {context.mark_node(make_list_construct(res))};
};

OutputVector translate_var_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto data = context.get_input(0);
    ov::Output<ov::Node> axes;

    if (!context.input_is_none(1)) {
        axes = context.get_input(1);
    }
    int32_t correction = 0;
    if (context.has_attribute("correction")) {
        correction = context.get_attribute<int32_t>("correction");
    }
    bool keepdim = false;
    if (context.has_attribute("keepdim")) {
        keepdim = context.get_attribute<bool>("keepdim");
    }
    auto res = translate_var_mean_common(context, data, axes, correction, keepdim);
    return {res[0]};
};

OutputVector translate_var(const NodeContext& context) {
    auto res = translate_var_mean(context);
    return {res[0]};
}

OutputVector translate_std_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto data = context.get_input(0);
    ov::Output<ov::Node> axes;

    if (!context.input_is_none(1)) {
        axes = context.get_input(1);
    }
    int32_t correction = 0;
    if (context.has_attribute("correction")) {
        correction = context.get_attribute<int32_t>("correction");
    }
    bool keepdim = false;
    if (context.has_attribute("keepdim")) {
        keepdim = context.get_attribute<bool>("keepdim");
    }
    auto res = translate_var_mean_common(context, data, axes, correction, keepdim);

    return {context.mark_node(std::make_shared<v0::Sqrt>(res[0]))};
}

OutputVector translate_std(const NodeContext& context) {
    auto res = translate_var_mean(context);
    auto var = res[0];
    return {context.mark_node(std::make_shared<v0::Sqrt>(var))};
}

OutputVector translate_std_mean(const NodeContext& context) {
    auto res = translate_var_mean(context);
    auto var = res[0];
    return {context.mark_node(std::make_shared<v0::Sqrt>(var)), res[1]};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
