// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/clip.hpp"

#include <limits>

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "validation_util.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector clip(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    const double max_value = node.get_attribute_value<double>("max", std::numeric_limits<double>::max());

    const double min_value = node.get_attribute_value<double>("min", std::numeric_limits<double>::lowest());

    return {std::make_shared<v0::Clamp>(data, min_value, max_value)};
}

}  // namespace set_1

namespace set_11 {
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

OutputVector clip(const Node& node) {
    const OutputVector inputs{node.get_ng_inputs()};
    const Output<ov::Node> data = inputs.at(0);
    const element::Type data_type = data.get_element_type();
    Output<ov::Node> min;
    Output<ov::Node> max;

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

}  // namespace set_11

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
