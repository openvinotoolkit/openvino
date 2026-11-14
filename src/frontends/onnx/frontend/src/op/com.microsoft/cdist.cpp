// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector cdist(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftcdist
    // Pairwise distances between the rows of A (M x N) and B (K x N), giving C (M x K). The Euclidean metrics are
    // expressed through broadcasting: C[i, j] = sum_k (A[i, k] - B[j, k])^2, followed by Sqrt for "euclidean".
    common::default_op_checks(node, 2, 2);

    const auto inputs = node.get_ov_inputs();
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto& a_type = a.get_element_type();
    CHECK_VALID_NODE(node,
                     a_type.is_dynamic() || a_type == ov::element::f32 || a_type == ov::element::f64,
                     "Unsupported input data type for A, expected FP32 or FP64 but got: ",
                     a_type);

    const auto metric = node.get_attribute_value<std::string>("metric", "euclidean");
    CHECK_VALID_NODE(node,
                     metric == "euclidean" || metric == "sqeuclidean",
                     "Only the 'euclidean' and 'sqeuclidean' metrics are supported for CDist. Got: ",
                     metric);

    // A: (M, N) -> (M, 1, N); B: (K, N) -> (1, K, N)
    const auto a_expanded = std::make_shared<v0::Unsqueeze>(a, v0::Constant::create(ov::element::i64, {}, {1}));
    const auto b_expanded = std::make_shared<v0::Unsqueeze>(b, v0::Constant::create(ov::element::i64, {}, {0}));
    const auto difference = std::make_shared<v1::Subtract>(a_expanded, b_expanded);
    const auto squared = std::make_shared<v1::Multiply>(difference, difference);
    const auto last_axis = v0::Constant::create(ov::element::i64, {}, {-1});
    const auto squared_distance = std::make_shared<v1::ReduceSum>(squared, last_axis, false);

    if (metric == "sqeuclidean") {
        return {squared_distance};
    }
    return {std::make_shared<v0::Sqrt>(squared_distance)};
}

ONNX_OP("CDist", OPSET_SINCE(1), com_microsoft::opset_1::cdist, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
