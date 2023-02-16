// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sort(NodeContext& context) {
    const auto input_tensor = context.get_input(0);
    const auto descending = context.const_input<bool>(2);
    // const auto stable = context.const_input<bool>(3);
    auto sort = ov::op::TopKSortType::SORT_VALUES;
    // if (stable) {
        // sort = ov::op::TopKSortType::SORT_INDICES;
    // }
    int64_t axis{-1};
    if (!context.input_is_none(1)) {
        axis = context.const_input<int64_t>(1);
    }
    auto mode = ov::op::TopKMode::MIN;
    if (descending) {
        mode = ov::op::TopKMode::MAX;
    }
    auto const_axis = context.mark_node(opset10::Constant::create(element::i32, {1}, {axis}));
    auto const_zero = context.mark_node(opset10::Constant::create(element::i32, {1}, {0}));
    auto shapeof = context.mark_node(std::make_shared<opset10::ShapeOf>(input_tensor));
    auto gather = context.mark_node(std::make_shared<opset10::Gather>(shapeof, const_axis, const_zero));
    auto squeeze = context.mark_node(std::make_shared<opset10::Squeeze>(gather, const_zero));

    auto topk = context.mark_node(std::make_shared<ov::op::v3::TopK>(input_tensor, squeeze, axis, mode, sort));
    auto indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(topk->output(1), element::i64));
    return {topk->output(0), indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov