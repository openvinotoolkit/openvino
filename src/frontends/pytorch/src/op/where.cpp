// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_where(NodeContext& context) {
    auto cond = context.get_input(0);
    if (context.input_is_none(1)) {
        auto non_zero_cond = context.mark_node(std::make_shared<opset8::NonZero>(cond, element::i64));
        auto unsqueezed_rank = context.mark_node(get_rank_node(cond));
        auto rank = context.mark_node(std::make_shared<opset8::Squeeze>(unsqueezed_rank));
        auto one = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
        auto zero = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
        auto split_lens = context.mark_node(std::make_shared<opset8::Broadcast>(one, unsqueezed_rank));
        auto tuple_results = std::make_shared<opset8::VariadicSplit>(non_zero_cond, zero, split_lens)->outputs();
        for (size_t i = 0; i < tuple_results.size(); i++) {
            tuple_results[i] = context.mark_node(std::make_shared<opset8::Squeeze>(tuple_results[i], zero));
        }
        return tuple_results;
    }
    auto bool_cond = context.mark_node(std::make_shared<opset8::Convert>(cond, element::boolean));
    auto x = context.get_input(1);
    auto y = context.get_input(2);
    return {context.mark_node(std::make_shared<opset8::Select>(bool_cond, x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov