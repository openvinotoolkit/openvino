// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_prod(const NodeContext& context) {
    // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
    num_inputs_check(context, 2, 4);
    auto input = context.get_input(0);
    bool keepdim;
    Output<Node> dim;
    int dtype_idx;
    if (context.get_input_size() == 2) {
        dim = get_axes_range(context, 0);
        keepdim = false;
        dtype_idx = 1;
    } else if (context.get_input_size() == 4) {
        dim = context.get_input(1);
        keepdim = context.const_input<bool>(2);
        dtype_idx = 3;
    } else {
        FRONT_END_GENERAL_CHECK(false, "Unexpected number of inputs.");
    }
    if (!context.input_is_none(dtype_idx)) {
        input = apply_dtype(context, dtype_idx, input);
    } else {
        // ReduceProd doesn't support boolean inputs
        auto data_dtype = simplified_type_interpret(context.get_input_type(0));
        if (input.get_element_type() == element::boolean ||
            (data_dtype.is<element::Type>() && data_dtype.as<element::Type>() == element::boolean)) {
            input = context.mark_node(std::make_shared<ov::op::v0::Convert>(input, element::i64));
        }
    }
    Output<Node> prod = context.mark_node(std::make_shared<ov::op::v1::ReduceProd>(input, dim, keepdim));
    return {prod};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
