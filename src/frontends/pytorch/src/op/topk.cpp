// Copyright (C) 2018-2025 Intel Corporation
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

OutputVector translate_topk(const NodeContext& context) {
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

    auto topk = context.mark_node(std::make_shared<v11::TopK>(input_tensor, k, axis, mode, sort, element::i64));

    return topk->outputs();
};

OutputVector translate_topk_fx(const NodeContext& context) {
    // aten.topk.default(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> Tuple[Tensor, Tensor]
    num_inputs_check(context, 2, 5);
    const auto input_tensor = context.get_input(0);
    auto k = context.get_input(1);
    int64_t axis{-1};
    bool largest = true;
    bool sorted = true;
    auto mode = TopKMode::MIN;
    auto sort = TopKSortType::NONE;

    if (!context.input_is_none(2)) {
        axis = context.const_input<int64_t>(2);
    }
    if (!context.input_is_none(3)) {
        largest = context.const_input<bool>(3);
    }
    if (!context.input_is_none(4)) {
        sorted = context.const_input<bool>(4);
    }
    if (largest) {
        mode = TopKMode::MAX;
    }
    if (sorted) {
        sort = TopKSortType::SORT_VALUES;
    }

    auto topk = context.mark_node(std::make_shared<v3::TopK>(input_tensor, k, axis, mode, sort));
    auto indices = context.mark_node(std::make_shared<v0::Convert>(topk->output(1), element::i64));

    return {context.mark_node(make_list_construct(OutputVector({topk->output(0), indices})))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
