// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/exp.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_expm1(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    // aten::expm1(Tensor self) -> Tensor
    // aten::expm1(Tensor self, Tensor out) -> Tensor
    auto input = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f32));

    auto exp = context.mark_node(std::make_shared<v0::Exp>(input));
    auto const_1 = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto expm1 = context.mark_node(std::make_shared<v1::Subtract>(exp, const_1));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, expm1);
    }
    return {expm1};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
