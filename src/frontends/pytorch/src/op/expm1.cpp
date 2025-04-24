// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_expm1(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    // aten::expm1(Tensor self) -> Tensor
    // aten::expm1(Tensor self, Tensor out) -> out Tensor

    auto exp_vector = translate_1to1_match_1_inputs_with_fp32_type_alignment<v0::Exp>(context);
    PYTORCH_OP_CONVERSION_CHECK(exp_vector.size() == 1,
                                "Expected exactly one element in the vector. Got: ",
                                exp_vector.size());
    auto exp = exp_vector[0];
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    const_1 = context.mark_node(std::make_shared<v1::ConvertLike>(const_1, exp));
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