// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector unique(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftunique
    // Outputs: y - the unique values of the 1D input in the order of their first occurrence,
    //          idx - for every input element the index of its value in y (int64),
    //          counts - the number of occurrences of every value of y (int64).
    // These are the first, third and fourth outputs of the unsorted ov::op::v10::Unique.
    common::default_op_checks(node, 1, 1);

    const auto& x = node.get_ov_inputs()[0];
    const bool sorted = false;
    const auto unique = std::make_shared<v10::Unique>(x, sorted, ov::element::i64, ov::element::i64);

    return {unique->output(0), unique->output(2), unique->output(3)};
}

ONNX_OP("Unique", OPSET_SINCE(1), com_microsoft::opset_1::unique, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
