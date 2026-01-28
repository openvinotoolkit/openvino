// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector where(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};

    const auto& then_input = ov_inputs.at(1);
    const auto& else_input = ov_inputs.at(2);

    const auto& then_type = then_input.get_element_type();
    const auto& else_type = else_input.get_element_type();

    // ONNX 'Where' operator allows different types for 'then' and 'else' branches in some cases (e.g. legacy models),
    // but OpenVINO 'Select' operator requires them to be of the same type.
    // We implement implicit type promotion here to handle this mismatch.
    if (then_type != else_type) {
        ov::element::Type target_type = then_type;
        // Promote to the larger bitwidth, or prefer Float > Signed > Unsigned if bitwidths are equal
        if (then_type.bitwidth() < else_type.bitwidth()) {
            target_type = else_type;
        } else if (then_type.bitwidth() == else_type.bitwidth()) {
            if (else_type.is_real() || (else_type.is_signed() && !then_type.is_real())) {
                target_type = else_type;
            }
        }

        if (then_type != target_type) {
            ov_inputs[1] = std::make_shared<ov::op::v0::Convert>(then_input, target_type);
        }
        if (else_type != target_type) {
            ov_inputs[2] = std::make_shared<ov::op::v0::Convert>(else_input, target_type);
        }
    }

    return {std::make_shared<ov::op::v1::Select>(ov_inputs.at(0), ov_inputs.at(1), ov_inputs.at(2))};
}
ONNX_OP("Where", OPSET_SINCE(1), ai_onnx::opset_1::where);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
