// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_max(const NodeContext& context) {
    // torch.max (same for torch.min) actually has two interfaces smashed together:
    // torch.max(x, dim, keepdim) and torch.max(x, y)
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    // torch.max(input)
    if (context.input_is_none(1) && context.input_is_none(2)) {
        auto axes = get_axes_range(context, 0);
        return {context.mark_node(std::make_shared<v1::ReduceMax>(x, axes, false))};
    }
    // torch.max(input, other)
    if (context.input_is_none(2)) {
        Output<Node> y;
        std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
        return {context.mark_node(std::make_shared<v1::Maximum>(x, y))};
    }
    // torch.max(input, dim, keepdim), returns values and indices
    auto axes_node = context.get_input(1);
    auto axis_const = context.const_input<int64_t>(1);
    auto keepdims = context.const_input<bool>(2);
    auto values = context.mark_node(std::make_shared<v1::ReduceMax>(x, axes_node, keepdims));
    auto k = context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto topk = context.mark_node(
        std::make_shared<v11::TopK>(x, k, axis_const, TopKMode::MAX, TopKSortType::NONE, element::i64));
    auto indices = topk->output(1);
    if (!keepdims) {
        indices = context.mark_node(std::make_shared<v0::Squeeze>(indices, axes_node));
    }
    return {values, indices};
};

OutputVector translate_max_dim(const NodeContext& context) {
    // torch.max.dim(x, dim, keepdim)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto axes_node = context.get_input(1);
    auto axis_const = context.const_input<int64_t>(1);

    bool keepdims = false;
    if (!context.input_is_none(2)) {
        keepdims = context.const_input<bool>(2);
    }

    auto values = context.mark_node(std::make_shared<v1::ReduceMax>(x, axes_node, keepdims));
    auto k = context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto topk = context.mark_node(
        std::make_shared<v11::TopK>(x, k, axis_const, TopKMode::MAX, TopKSortType::NONE, element::i64));
    auto indices = topk->output(1);
    if (!keepdims) {
        indices = context.mark_node(std::make_shared<v0::Squeeze>(indices, axes_node));
    }
    return {values, indices};
};

OutputVector translate_max_dim_fx(const NodeContext& context) {
    ov::OutputVector out_vec = translate_max_dim(context);
    return {context.mark_node(make_list_construct(out_vec))};
};

OutputVector translate_min(const NodeContext& context) {
    // torch.min (same for torch.max) actually has two interfaces smashed together:
    // torch.min(x, dim, keepdim) and torch.min(x, y)
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    // torch.min(input)
    if (context.input_is_none(1) && context.input_is_none(2)) {
        auto axes = get_axes_range(context, 0);
        return {context.mark_node(std::make_shared<v1::ReduceMin>(x, axes, false))};
    }
    // torch.min(input, other)
    if (context.input_is_none(2)) {
        Output<Node> y;
        std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
        return {context.mark_node(std::make_shared<v1::Minimum>(x, y))};
    }
    // torch.min(input, dim, keepdim), returns values and indices
    auto axes_node = context.get_input(1);
    auto axis_const = context.const_input<int64_t>(1);
    auto keepdims = context.const_input<bool>(2);
    auto values = context.mark_node(std::make_shared<v1::ReduceMin>(x, axes_node, keepdims));
    auto k = context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto topk = context.mark_node(
        std::make_shared<v11::TopK>(x, k, axis_const, TopKMode::MIN, TopKSortType::NONE, element::i64));
    auto indices = topk->output(1);
    if (!keepdims) {
        indices = context.mark_node(std::make_shared<v0::Squeeze>(indices, axes_node));
    }
    return {values, indices};
};

OutputVector translate_min_dim(const NodeContext& context) {
    // torch.min.dim(x, dim, keepdim)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto axes_node = context.get_input(1);
    auto axis_const = context.const_input<int64_t>(1);

    bool keepdims = false;
    if (!context.input_is_none(2)) {
        keepdims = context.const_input<bool>(2);
    }

    auto values = context.mark_node(std::make_shared<v1::ReduceMin>(x, axes_node, keepdims));
    auto k = context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto topk = context.mark_node(
        std::make_shared<v11::TopK>(x, k, axis_const, TopKMode::MIN, TopKSortType::NONE, element::i64));
    auto indices = topk->output(1);
    if (!keepdims) {
        indices = context.mark_node(std::make_shared<v0::Squeeze>(indices, axes_node));
    }
    return {values, indices};
};

OutputVector translate_min_dim_fx(const NodeContext& context) {
    ov::OutputVector out_vec = translate_min_dim(context);
    return {context.mark_node(make_list_construct(out_vec))};
};

OutputVector translate_maximum(const NodeContext& context) {
    // aten::maximum(Tensor self, Tensor other) -> Tensor

    //  aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

    num_inputs_check(context, 2, 3);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    auto res = context.mark_node(std::make_shared<v1::Maximum>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, res);
    }
    return {res};
}

OutputVector translate_minimum(const NodeContext& context) {
    // aten::minimum(Tensor self, Tensor other) -> Tensor

    //  aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

    num_inputs_check(context, 2, 3);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    auto res = context.mark_node(std::make_shared<v1::Minimum>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, res);
    }
    return {res};
}

OutputVector translate_amin(const NodeContext& context) {
    // aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor

    // aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto dims = context.get_input(1);
    bool keep_dims = false;
    if (!context.input_is_none(2)) {
        keep_dims = context.const_input<bool>(2);
    }
    auto res = context.mark_node(std::make_shared<v1::ReduceMin>(x, dims, keep_dims));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, res);
    }
    return {res};
}

OutputVector translate_amax(const NodeContext& context) {
    // aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor

    // aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto dims = context.get_input(1);
    bool keep_dims = false;
    if (!context.input_is_none(2)) {
        keep_dims = context.const_input<bool>(2);
    }
    auto res = context.mark_node(std::make_shared<v1::ReduceMax>(x, dims, keep_dims));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, res);
    }
    return {res};
}

OutputVector translate_aminmax(const NodeContext& context) {
    num_inputs_check(context, 1, 4);  // Expect between 1 and 4 inputs
                                      // (input tensor, dim = none, keepdim = false, out = none)

    auto input = context.get_input(0);

    // check if dim is provided, if not, get the range of axes to compute min and max
    auto dim = !context.input_is_none(1) ? context.get_input(1) : get_axes_range(context, 0);

    // check if keepdim is provided, if not, set it to false like PyTorch
    bool keep_dims = !context.input_is_none(2) ? context.const_input<bool>(2) : false;

    auto amin = context.mark_node(std::make_shared<v1::ReduceMin>(input, dim, keep_dims));
    auto amax = context.mark_node(std::make_shared<v1::ReduceMax>(input, dim, keep_dims));

    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(3), "out argument is not supported for aten::aminmax");

    return {amin, amax};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
