// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector reduce_sum_integer(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftreducesuminteger
    // Sums an 8-bit integer tensor along the given axes into a 32-bit result: INT8 -> INT32, UINT8 -> UINT32.
    common::default_op_checks(node, 1, 1);

    const auto& data = node.get_ov_inputs()[0];
    const auto& data_type = data.get_element_type();
    CHECK_VALID_NODE(node,
                     data_type == ov::element::u8 || data_type == ov::element::i8,
                     "Unsupported input data type, expected INT8 or UINT8 but got: ",
                     data_type);
    CHECK_VALID_NODE(node, node.has_attribute("axes"), "Required attribute 'axes' is not specified.");

    const auto axes = node.get_attribute_as_constant<std::vector<int64_t>>("axes");
    const bool keep_dims = node.get_attribute_value<int64_t>("keepdims", 1) != 0;
    const auto result_type = data_type == ov::element::u8 ? ov::element::u32 : ov::element::i32;

    const auto converted_data = std::make_shared<v0::Convert>(data, result_type);
    return {std::make_shared<v1::ReduceSum>(converted_data, axes, keep_dims)};
}

ONNX_OP("ReduceSumInteger", OPSET_SINCE(1), com_microsoft::opset_1::reduce_sum_integer, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
