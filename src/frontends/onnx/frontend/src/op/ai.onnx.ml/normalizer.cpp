// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cmath>
#include <numeric>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx_ml {
namespace opset_1 {

const ov::Output<ov::Node> check_zero_divisor(const ov::Output<ov::Node>& val) {
    const auto zero = ov::op::v0::Constant::create(val.get_element_type(), ov::Shape{}, {0.0f});
    const auto one_val = ov::op::v0::Constant::create(val.get_element_type(), ov::Shape{}, {1.0f});
    const auto is_zero = std::make_shared<ov::op::v1::Equal>(val, zero);
    return std::make_shared<ov::op::v1::Select>(is_zero, one_val, val);
}

ov::OutputVector normalizer(const ov::frontend::onnx::Node& node) {
    // Step 1: Get input tensor
    const auto input = node.get_ov_inputs()[0];

    CHECK_VALID_NODE(node,
                     input.get_element_type() == ov::element::f64 || input.get_element_type() == ov::element::f32 ||
                         input.get_element_type() == ov::element::i64 || input.get_element_type() == ov::element::i32,
                     "Unsupported input type, accepted FP32, FP64, I64, I32 got: ",
                     input.get_element_type());

    const auto normalization_type = node.get_attribute_value<std::string>("norm", "MAX");

    CHECK_VALID_NODE(node,
                     normalization_type == "MAX" || normalization_type == "L1" || normalization_type == "L2",
                     "Normalization Mode should be either MAX, L1 or L2, got ",
                     normalization_type);

    auto float_input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    const auto x = std::make_shared<ov::op::v0::Abs>(float_input);
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});  // e.g., last axis
    if (normalization_type == "L1") {
        const auto sum_x = std::make_shared<ov::op::v1::ReduceSum>(x, axes, true);
        const auto safe_x = check_zero_divisor(sum_x);
        return {std::make_shared<ov::op::v1::Divide>(float_input, safe_x)};
    } else if (normalization_type == "L2") {
        const auto x_squared = std::make_shared<ov::op::v1::Multiply>(x, x);
        const auto sum_sqrdx = std::make_shared<ov::op::v1::ReduceSum>(x_squared, axes, true);
        const auto sqrt_sum_sqrdx = std::make_shared<ov::op::v0::Sqrt>(sum_sqrdx);
        const auto safe_x = check_zero_divisor(sqrt_sum_sqrdx);
        const auto result = std::make_shared<ov::op::v1::Divide>(float_input, safe_x);
        return {result};
    } else {  // Must be max
        const auto max_x = std::make_shared<ov::op::v1::ReduceMax>(x, axes, /*keep_dims=*/true);
        const auto safe_x = check_zero_divisor(max_x);
        return {std::make_shared<ov::op::v1::Divide>(float_input, safe_x)};
    }
}

ONNX_OP("Normalizer", OPSET_SINCE(1), ai_onnx_ml::opset_1::normalizer, AIONNX_ML_DOMAIN);
}  // namespace opset_1
}  // namespace ai_onnx_ml
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
