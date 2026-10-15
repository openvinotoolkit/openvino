// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_type_as(const NodeContext& context) {
    // aten::type_as(Tensor self, Tensor other) -> Tensor
    // Converts self to the dtype of other
    num_inputs_check(context, 2, 2, true);  // allow_complex = true
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");

    auto input = context.get_input(0);
    auto like = context.get_input(1);

    // Use ComplexTypeMark::convert_like for proper handling of complex types
    return {ComplexTypeMark::convert_like(context, input, like)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
