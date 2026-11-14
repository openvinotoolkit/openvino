// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector expand_dims(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftexpanddims
    // Inserts a dimension of size 1 at the position given by the `axis` input (negative values count from the
    // end of the output shape), which is exactly what Unsqueeze does.
    common::default_op_checks(node, 2, 2);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];
    const auto& axis = inputs[1];
    const auto& axis_type = axis.get_element_type();
    CHECK_VALID_NODE(node,
                     axis_type.is_dynamic() || axis_type.is_integral_number(),
                     "Unsupported data type for axis, expected an integer tensor but got: ",
                     axis_type);

    return {std::make_shared<v0::Unsqueeze>(x, axis)};
}

ONNX_OP("ExpandDims", OPSET_SINCE(1), com_microsoft::opset_1::expand_dims, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
