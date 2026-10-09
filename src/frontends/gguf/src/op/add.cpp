// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_add(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    const auto lhs_type = lhs.get_element_type();
    const auto rhs_type = rhs.get_element_type();
    const auto output_type = context.get_attribute<ov::element::Type>("output_type", lhs_type);
    if (lhs_type == rhs_type && output_type == lhs_type) {
        return rename_outputs_with_suffix({std::make_shared<ov::op::v1::Add>(lhs, rhs)}, context.get_name());
    }
    const auto is_float = [](const ov::element::Type& type) {
        return type == ov::element::f32 || type == ov::element::f16 || type == ov::element::bf16;
    };
    FRONT_END_OP_CONVERSION_CHECK(is_float(lhs_type) && is_float(rhs_type) && is_float(output_type),
                                  "Unsupported ADD input/output types: ", lhs_type, ", ", rhs_type, ", ", output_type);
    // GGML accumulates mixed floating-point operands in f32, then rounds to the destination type.
    if (lhs_type != ov::element::f32) {
        lhs = std::make_shared<ov::op::v0::Convert>(lhs, ov::element::f32);
    }
    if (rhs_type != ov::element::f32) {
        rhs = std::make_shared<ov::op::v0::Convert>(rhs, ov::element::f32);
    }
    ov::Output<Node> result = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    if (output_type != ov::element::f32) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
