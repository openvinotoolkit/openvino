// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/common/random_normal_helper.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector random_normal(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node, node.has_attribute("shape"), "RandomNormal operator must specify a 'shape' attribute.");

    const auto dtype =
        node.get_attribute_value<int64_t>("dtype",
                                          static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT));
    const auto target_type = common::get_ov_element_type(dtype);

    const auto mean = node.get_attribute_value<float>("mean", 0.0f);
    const auto scale = node.get_attribute_value<float>("scale", 1.0f);
    auto scale_node = v0::Constant::create(target_type, ov::Shape{1}, {scale});
    auto mean_node = v0::Constant::create(target_type, ov::Shape{1}, {mean});

    const auto seed = node.get_attribute_value<float>("seed", 0);
    const auto shape = node.get_attribute_as_constant<std::vector<int64_t>>("shape");
    auto res = ov::frontend::make_random_normal(shape, target_type, mean_node, scale_node, seed);
    return res.first;
}

ONNX_OP("RandomNormal", OPSET_SINCE(1), ai_onnx::opset_1::random_normal);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
