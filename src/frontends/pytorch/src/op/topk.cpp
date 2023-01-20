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

OutputVector translate_topk(NodeContext& context) {
    const auto input_tensor = context.get_input(0);
    const auto largest = context.const_input<bool>(3);
    const auto sorted = context.const_input<bool>(4);
    auto k = context.get_input(1);
    int64_t axis{-1};
    auto mode = opset10::TopK::Mode::MAX;
    auto sort = opset10::TopK::SortType::SORT_VALUES;

    if (!context.input_is_none(2)) {
        axis = context.const_input<int64_t>(2);
    }
    if (!largest) {
        mode = opset10::TopK::Mode::MIN;
    }
    if (!sorted) {
        sort = opset10::TopK::SortType::NONE;
    }

    auto topk = context.mark_node(std::make_shared<opset10::TopK>(input_tensor, k, axis, mode, sort));
    auto indices = context.mark_node(std::make_shared<opset10::Convert>(topk->output(1), element::i64));

    return {topk->output(0), indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
