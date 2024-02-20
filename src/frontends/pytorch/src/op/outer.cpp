// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_outer(const NodeContext& context) {
    // aten::outer(Tensor self, Tensor vec2) -> Tensor
    // aten::outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 2, 3);
    auto vec1 = context.get_input(0);
    auto vec2 = context.get_input(1);
    align_eltwise_input_types(context, vec1, vec2, true);
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    vec1 = context.mark_node(std::make_shared<v0::Unsqueeze>(vec1, const_minus_one));
    vec2 = context.mark_node(std::make_shared<v0::Unsqueeze>(vec2, const_zero));
    auto out = context.mark_node(std::make_shared<v0::MatMul>(vec1, vec2));
    if (!context.input_is_none(2)) {
        out = context.mark_node(std::make_shared<v1::ConvertLike>(out, context.get_input(2)));
        context.mutate_input(2, out);
    }
    return {out};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov