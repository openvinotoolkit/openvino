// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/gather.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/unsqueeze.hpp>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_kthvalue(const NodeContext& context) {
    // aten::kthvalue(Tensor input, int k, int dim=None, bool keepdim) -> (Tensor values, Tensor indices)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto k = context.get_input(1);

    int64_t dim = -1;
    if (!context.input_is_none(2)) {
        dim = context.const_input<int64_t>(2);
    }

    bool keepdim = false;
    if (!context.input_is_none(3)) {
        keepdim = context.const_input<bool>(3);
    }

    auto axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));

    auto smallest_k = context.mark_node(
        std::make_shared<v11::TopK>(x, k, dim, v3::TopK::Mode::MIN, v3::TopK::SortType::SORT_VALUES, element::i64));

    auto k_minus_1 =
        context.mark_node(std::make_shared<v1::Subtract>(k, v0::Constant::create(k.get_element_type(), {}, {1})));

    auto kth_value = context.mark_node(std::make_shared<v8::Gather>(smallest_k->output(0), k_minus_1, axis_const));
    auto kth_index = context.mark_node(std::make_shared<v8::Gather>(smallest_k->output(1), k_minus_1, axis_const));

    if (keepdim) {
        kth_value = context.mark_node(std::make_shared<v0::Unsqueeze>(kth_value, axis_const));
        kth_index = context.mark_node(std::make_shared<v0::Unsqueeze>(kth_index, axis_const));
    }

    return {kth_value, kth_index};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov