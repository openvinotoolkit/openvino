// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector less_or_equal(const ov::frontend::onnx::Node& node) {
    const auto& input = node.get_ov_inputs();
    const auto a = input.at(0);
    const auto b = input.at(1);
    FRONT_END_GENERAL_CHECK(a.get_element_type() != ov::element::bf16 && b.get_element_type() != ov::element::bf16,
                            "The input data bfloat16 isn't supported in opset 12");
    return {std::make_shared<v1::LessEqual>(a, b)};
}
static bool registered = register_translator("LessOrEqual", VersionRange{1, 15}, less_or_equal);
}  // namespace opset_1

namespace opset_16 {
ov::OutputVector less_or_equal(const ov::frontend::onnx::Node& node) {
    const auto& input = node.get_ov_inputs();
    const auto a = input.at(0);
    const auto b = input.at(1);
    return {std::make_shared<v1::LessEqual>(a, b)};
}
static bool registered = register_translator("LessOrEqual", VersionRange::since(16), less_or_equal);
}  // namespace opset_16
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
