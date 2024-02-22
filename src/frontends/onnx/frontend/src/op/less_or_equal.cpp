// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/less_or_equal.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector less_or_equal(const ov::frontend::onnx::Node& node) {
    const auto& input = node.get_ov_inputs();
    const auto a = input.at(0);
    const auto b = input.at(1);
    FRONT_END_GENERAL_CHECK(a.get_element_type() != ov::element::bf16 && b.get_element_type() != ov::element::bf16,
                            "The input data bfloat16 isn't supported in opset 12");
    return {std::make_shared<v1::LessEqual>(a, b)};
}
}  // namespace set_1

namespace set_16 {
ov::OutputVector less_or_equal(const ov::frontend::onnx::Node& node) {
    const auto& input = node.get_ov_inputs();
    const auto a = input.at(0);
    const auto b = input.at(1);
    return {std::make_shared<v1::LessEqual>(a, b)};
}
}  // namespace set_16
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
