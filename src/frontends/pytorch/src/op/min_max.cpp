// Copyright (C) 2018-2023 Intel Corporation
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
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_max(NodeContext& context) {
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
        auto y = context.get_input(1);
        return {context.mark_node(std::make_shared<v1::Maximum>(x, y))};
    }
    // torch.max(input, dim, keepdim), returns values and indicies
    auto axes_node = context.get_input(1);
    auto axis_const = context.const_input<int64_t>(1);
    auto keepdims = context.const_input<bool>(2);
    auto values = context.mark_node(std::make_shared<v1::ReduceMax>(x, axes_node, keepdims));
    auto k = context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto topk = std::make_shared<v3::TopK>(x, k, axis_const, v3::TopK::Mode::MAX, v3::TopK::SortType::NONE);
    auto indicies = context.mark_node(std::make_shared<v0::Convert>(topk->output(1), element::i64));
    if (!keepdims) {
        indicies = std::make_shared<v0::Squeeze>(indicies, axes_node);
    }
    return {values, indicies};
};

OutputVector translate_min(NodeContext& context) {
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
        auto y = context.get_input(1);
        return {context.mark_node(std::make_shared<v1::Minimum>(x, y))};
    }
    // torch.min(input, dim, keepdim), returns values and indicies
    auto axes_node = context.get_input(1);
    auto axis_const = context.const_input<int64_t>(1);
    auto keepdims = context.const_input<bool>(2);
    auto values = context.mark_node(std::make_shared<v1::ReduceMin>(x, axes_node, keepdims));
    auto k = context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto topk = std::make_shared<v3::TopK>(x, k, axis_const, v3::TopK::Mode::MIN, v3::TopK::SortType::NONE);
    auto indicies = context.mark_node(std::make_shared<v0::Convert>(topk->output(1), element::i64));
    if (!keepdims) {
        indicies = std::make_shared<v0::Squeeze>(indicies, axes_node);
    }
    return {values, indicies};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov