// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_diff(const NodeContext& context) {
    // aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
    num_inputs_check(context, 1, 5);
    auto x = context.get_input(0);
    const int64_t n = context.input_is_none(1) ? 1 : context.const_input<int64_t>(1);
    const int64_t dim = context.input_is_none(2) ? -1 : context.const_input<int64_t>(2);
    if (!context.input_is_none(3) || !context.input_is_none(4)) {
        OutputVector parts;
        if (!context.input_is_none(3))
            parts.push_back(context.get_input(3));
        parts.push_back(x);
        if (!context.input_is_none(4))
            parts.push_back(context.get_input(4));
        x = context.mark_node(std::make_shared<v0::Concat>(parts, dim));
    }
    const auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    const auto one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    const auto minus_one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    const auto int_max = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {INT_MAX}));
    const auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {dim}));
    for (int64_t i = 0; i < n; ++i) {
        const auto next = context.mark_node(std::make_shared<v8::Slice>(x, one, int_max, one, axis));
        const auto prev = context.mark_node(std::make_shared<v8::Slice>(x, zero, minus_one, one, axis));
        // PyTorch computes diff of bool tensors as logical xor.
        if (x.get_element_type() == element::boolean) {
            x = context.mark_node(std::make_shared<v1::NotEqual>(next, prev));
        } else {
            x = context.mark_node(std::make_shared<v1::Subtract>(next, prev));
        }
    }
    return {x};
};

}  // namespace ov::frontend::pytorch::op
