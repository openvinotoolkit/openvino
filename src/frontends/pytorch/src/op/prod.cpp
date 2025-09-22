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

using namespace ov::op;

Output<Node> translate_prod_common(const NodeContext& context,
                                   const Output<Node>& input,
                                   const Output<Node>& dim,
                                   bool keepdim,
                                   bool skip_bool_check) {
    // ReduceProd doesn't support boolean inputs
    auto input_tensor = input;
    if (!skip_bool_check) {
        auto data_dtype = simplified_type_interpret(context.get_input_type(0));
        if (input_tensor.get_element_type() == element::boolean ||
            (data_dtype.is<element::Type>() && data_dtype.as<element::Type>() == element::boolean)) {
            input_tensor = context.mark_node(std::make_shared<ov::op::v0::Convert>(input_tensor, element::i64));
        }
    }
    return context.mark_node(std::make_shared<ov::op::v1::ReduceProd>(input_tensor, dim, keepdim));
}

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
    bool skip_bool_check = false;
    if (!context.input_is_none(dtype_idx)) {
        skip_bool_check = true;
        input = apply_dtype(context, dtype_idx, input);
    }
    auto prod = translate_prod_common(context, input, dim, keepdim, skip_bool_check);
    return {prod};
};

OutputVector translate_prod_fx(const NodeContext& context) {
    // aten.prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    // aten.prod(Tensor self, *, ScalarType? dtype=None) -> Tensor

    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    bool keepdim = false;
    Output<Node> dim;
    if (!context.input_is_none(1)) {
        dim = context.get_input(1);
    } else {
        dim = get_axes_range(context, 0);
    }
    if (!context.input_is_none(2)) {
        keepdim = context.const_input<bool>(2);
    }
    bool skip_bool_check = false;
    if (context.has_attribute("dtype")) {
        skip_bool_check = true;
        auto dtype = context.get_attribute<element::Type>("dtype");
        input = context.mark_node(std::make_shared<ov::op::v0::Convert>(input, dtype));
    }
    auto prod = translate_prod_common(context, input, dim, keepdim, skip_bool_check);
    return {prod};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
