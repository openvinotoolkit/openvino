// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cmath"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

std::vector<size_t> get_transpose_axes(size_t rank) {
    std::vector<size_t> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);

    if (rank > 2) {
        std::rotate(axes.begin() + 1, axes.begin() + 2, axes.begin() + rank - 1);
    }
    return axes;
}

ov::OutputVector fusedmatmul(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    const auto inputs = node.get_ov_inputs();
    auto A = inputs[0];  // required
    auto B = inputs[1];  // required

    const auto alpha = node.get_attribute_value<float>("alpha", 1.f);
    const auto transA = node.get_attribute_value<int64_t>("transA", 0);
    const auto transB = node.get_attribute_value<int64_t>("transB", 0);
    const auto transBatchA = node.get_attribute_value<int64_t>("transBatchA", 0);
    const auto transBatchB = node.get_attribute_value<int64_t>("transBatchB", 0);

    CHECK_VALID_NODE(node,
                     A.get_element_type() == ov::element::f16 || A.get_element_type() == ov::element::f32 ||
                         A.get_element_type() == ov::element::f64 || A.get_element_type() == ov::element::bf16,
                     "Unsupported input A type, accepted FP16, FP32, FP64, BFP16 got: ",
                     A.get_element_type());
    CHECK_VALID_NODE(node,
                     B.get_element_type() == ov::element::f16 || B.get_element_type() == ov::element::f32 ||
                         B.get_element_type() == ov::element::f64 || B.get_element_type() == ov::element::bf16,
                     "Unsupported input B type, accepted FP16, FP32, FP64, BFP16 got: ",
                     B.get_element_type());

    const auto rankA = A.get_partial_shape().rank();
    const auto rankB = B.get_partial_shape().rank();

    if (transBatchA && rankA.is_static()) {
        auto rank = rankA.get_length();
        A = std::make_shared<v1::Transpose>(
            A,
            std::make_shared<v0::Constant>(element::i64, Shape{static_cast<size_t>(rank)}, get_transpose_axes(rank)));
    }

    if (transBatchB && rankB.is_static()) {
        auto rank = rankB.get_length();
        B = std::make_shared<v1::Transpose>(
            B,
            std::make_shared<v0::Constant>(element::i64, Shape{static_cast<size_t>(rank)}, get_transpose_axes(rank)));
    }

    auto matmul_result = std::make_shared<v0::MatMul>(A, B, transA, transB);

    auto alpha_const = std::make_shared<v0::Constant>(A.get_element_type(), Shape{}, std::vector<float>{alpha});
    auto scaled_result = std::make_shared<v1::Multiply>(matmul_result, alpha_const);

    return {scaled_result};
}

ONNX_OP("FusedMatMul", OPSET_SINCE(1), com_microsoft::opset_1::fusedmatmul, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
