// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_15 {
namespace {
const std::set<element::Type> supported_input_types = {element::f16, element::bf16, element::f32, element::f64};
const std::set<element::Type> supported_output_types = {element::u8,
                                                        element::u16,
                                                        element::u32,
                                                        element::u64,
                                                        element::i8,
                                                        element::i16,
                                                        element::i32,
                                                        element::i64,
                                                        element::f16,
                                                        element::bf16,
                                                        element::f32,
                                                        element::f64,
                                                        element::boolean};
}  // namespace

ov::OutputVector bernoulli(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 1, 1);
    const auto input = node.get_ov_inputs().at(0);
    const auto& input_type = input.get_element_type();
    CHECK_VALID_NODE(node,
                     input_type.is_dynamic() || supported_input_types.count(input_type) > 0,
                     "Unsupported input type ",
                     input_type.get_type_name());

    // RandomUniform needs a static type, so f32 samples are used when the input type is unknown.
    const auto sample_type = input_type.is_dynamic() ? element::f32 : input_type;
    const auto shape = std::make_shared<v3::ShapeOf>(input);
    const auto low = v0::Constant::create(sample_type, Shape{}, {0});
    const auto high = v0::Constant::create(sample_type, Shape{}, {1});
    const auto seed = common::convert_float_seed(node.get_attribute_value<float>("seed", 0.0f));
    const uint64_t global_seed = 0;
    ov::Output<ov::Node> samples =
        std::make_shared<v8::RandomUniform>(shape, low, high, sample_type, global_seed, seed);
    if (input_type.is_dynamic()) {
        samples = std::make_shared<v1::ConvertLike>(samples, input);
    }

    // P(U < p) == p for U ~ Uniform[0, 1).
    const auto mask = std::make_shared<v1::Less>(samples, input);

    if (node.has_attribute("dtype")) {
        const auto& output_type = common::get_ov_element_type(node.get_attribute_value<int64_t>("dtype"));
        CHECK_VALID_NODE(node,
                         supported_output_types.count(output_type) > 0,
                         "Unsupported dtype ",
                         output_type.get_type_name());
        return {std::make_shared<v0::Convert>(mask, output_type)};
    }
    return {std::make_shared<v1::ConvertLike>(mask, input)};
}

ONNX_OP("Bernoulli", OPSET_SINCE(1), ai_onnx::opset_15::bernoulli);
}  // namespace ov::frontend::onnx::ai_onnx::opset_15
