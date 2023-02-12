// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/topk.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"

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
    auto mode = ov::op::TopKMode::MIN;
    auto sort = ov::op::TopKSortType::NONE;

    if (!context.input_is_none(2)) {
        axis = context.const_input<int64_t>(2);
    }
    if (largest) {
        mode = ov::op::TopKMode::MAX;
    }
    if (sorted) {
        sort = ov::op::TopKSortType::SORT_VALUES;
    }

    auto topk = context.mark_node(std::make_shared<ov::op::v3::TopK>(input_tensor, k, axis, mode, sort));
    auto indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(topk->output(1), element::i64));

    return {topk->output(0), indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
