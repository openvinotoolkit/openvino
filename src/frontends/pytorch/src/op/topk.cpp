// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/topk.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_topk(NodeContext& context) {
    num_inputs_check(context, 5, 5);
    const auto input_tensor = context.get_input(0);
    const auto largest = context.const_input<bool>(3);
    const auto sorted = context.const_input<bool>(4);
    auto k = context.get_input(1);
    int64_t axis{-1};
    auto mode = TopKMode::MIN;
    auto sort = TopKSortType::NONE;

    if (!context.input_is_none(2)) {
        axis = context.const_input<int64_t>(2);
    }
    if (largest) {
        mode = TopKMode::MAX;
    }
    if (sorted) {
        sort = TopKSortType::SORT_VALUES;
    }

    auto topk = context.mark_node(std::make_shared<v3::TopK>(input_tensor, k, axis, mode, sort));
    auto indices = context.mark_node(std::make_shared<v0::Convert>(topk->output(1), element::i64));

    return {topk->output(0), indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
