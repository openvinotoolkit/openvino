// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector clip(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    const double max_value = node.get_attribute_value<double>("max", std::numeric_limits<double>::max());

    const double min_value = node.get_attribute_value<double>("min", std::numeric_limits<double>::lowest());

    return {std::make_shared<v0::Clamp>(data, min_value, max_value)};
}

ONNX_OP("Clip", OPSET_RANGE(1, 10), ai_onnx::opset_1::clip);
}  // namespace opset_1

namespace opset_11 {
namespace {
std::shared_ptr<ov::op::v0::Constant> get_constant_lowest_of_type(ov::element::Type_t t) {
#define OPENVINO_TYPE_TO_LOWEST_CONST(t)                                                       \
    case t:                                                                                    \
        return ov::op::v0::Constant::create(                                                   \
            t,                                                                                 \
            {},                                                                                \
            {std::numeric_limits<typename ov::element_type_traits<t>::value_type>::lowest()}); \
        break

    switch (t) {
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::boolean);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::bf16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::f16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::f32);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::f64);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i8);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i32);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i64);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u1);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u8);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u32);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u64);

    case ov::element::undefined:
    case ov::element::dynamic:
    default:
        return nullptr;
    }
}

std::shared_ptr<ov::op::v0::Constant> get_constant_max_of_type(ov::element::Type_t t) {
    auto tensor = ov::util::make_tensor_of_max_value(t);
    return tensor ? std::make_shared<ov::op::v0::Constant>(tensor) : nullptr;
}
}  // namespace

ov::OutputVector clip(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector inputs{node.get_ov_inputs()};
    const ov::Output<ov::Node> data = inputs.at(0);
    const ov::element::Type data_type = data.get_element_type();
    ov::Output<ov::Node> min;
    ov::Output<ov::Node> max;

    // If second input is provided, assign to min input, otherwise set lowest
    // numeric limit of data type as min input.
    if (inputs.size() > 1 && !ov::op::util::is_null(inputs.at(1))) {
        min = inputs.at(1);
    } else {
        min = get_constant_lowest_of_type(data_type);
    }

    // If third input is provided, assign to max input, otherwise set maximum
    // numeric limit of data type as max input.
    if (inputs.size() == 3 && !ov::op::util::is_null(inputs.at(2))) {
        max = inputs.at(2);
    } else {
        max = get_constant_max_of_type(data_type);
    }

    const auto max_of_min_and_data = std::make_shared<v1::Maximum>(min, data);

    return {std::make_shared<v1::Minimum>(max, max_of_min_and_data)};
}

ONNX_OP("Clip", OPSET_SINCE(11), ai_onnx::opset_11::clip);
}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
