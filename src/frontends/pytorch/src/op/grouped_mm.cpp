// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// torch._grouped_mm(Tensor self, Tensor mat2,
//                   Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
//
// Maps directly to ov::op::v17::GroupedMatMul, which natively supports the
// 3D x 3D batched case as well as the 2D x 3D and 2D x 2D MoE cases (the latter
// two via `offs`).
OutputVector translate_grouped_mm(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto a = context.get_input(0);
    auto b = context.get_input(1);

    const bool has_offs = context.get_input_size() > 2 && !context.input_is_none(2);
    const bool has_bias = context.get_input_size() > 3 && !context.input_is_none(3);
    const bool has_out_dtype = context.get_input_size() > 4 && !context.input_is_none(4);

    PYTORCH_OP_CONVERSION_CHECK(!has_bias, "_grouped_mm: 'bias' argument is not supported.");

    align_eltwise_input_types(context, a, b, false, false);

    // PyTorch provides mat_b as [G, K, N]; GroupedMatMul-17 expects [G, N, K].
    // Apply the transpose only for 3D weights (Cases 1 and 2); leave 2D untouched (Case 3).
    const auto b_rank = b.get_partial_shape().rank();
    if (b_rank.is_static() && b_rank.get_length() == 3) {
        auto perm = context.mark_node(v0::Constant::create(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1}));
        b = context.mark_node(std::make_shared<v1::Transpose>(b, perm));
    }

    Output<Node> result;
    if (has_offs) {
        auto offs = context.get_input(2);
        // GroupedMatMul expects an integer offsets tensor; PyTorch typically
        // provides int32, but normalize anything else to i32 to be safe.
        if (offs.get_element_type().is_static() && offs.get_element_type() != element::i32 &&
            offs.get_element_type() != element::i64) {
            offs = context.mark_node(std::make_shared<v0::Convert>(offs, element::i32));
        }
        result = context.mark_node(std::make_shared<v17::GroupedMatMul>(a, b, offs));
    } else {
        result = context.mark_node(std::make_shared<v17::GroupedMatMul>(a, b));
    }

    if (has_out_dtype) {
        result = apply_dtype(context, 4, result);
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
