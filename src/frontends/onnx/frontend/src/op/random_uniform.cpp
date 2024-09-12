// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_uniform.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector random_uniform(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node, node.has_attribute("shape"), "RandomUniform operator must specify a 'shape' attribute.");

    const auto dtype =
        node.get_attribute_value<int64_t>("dtype",
                                          static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT));
    const auto high_const = node.get_attribute_as_constant<float>("high", 1.0f);
    const auto low_const = node.get_attribute_as_constant<float>("low", 0.0f);
    const auto seed = common::convert_float_seed(node.get_attribute_value<float>("seed", 0.0f));
    const auto target_shape_const = node.get_attribute_as_constant<std::vector<int64_t>>("shape");

    const auto target_type = common::get_ov_element_type(dtype);
    const uint64_t global_seed = 0;

    return {
        std::make_shared<v8::RandomUniform>(target_shape_const, low_const, high_const, target_type, global_seed, seed)};
}

ONNX_OP("RandomUniform", OPSET_SINCE(1), ai_onnx::opset_1::random_uniform);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
