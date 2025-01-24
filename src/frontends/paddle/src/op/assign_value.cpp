// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"
namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs assign_value(const NodeContext& node) {
    std::vector<int32_t> shape = node.get_attribute<std::vector<int32_t>>("shape");
    auto dtype = node.get_attribute<ov::element::Type>("dtype");
    std::shared_ptr<Node> const_node;

    switch (dtype) {
    case element::i32: {
        if (node.has_attribute("int32_values")) {
            auto values = node.get_attribute<std::vector<int32_t>>("int32_values");
            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        } else {
            auto values = node.get_attribute<std::vector<int64_t>>("values");
            std::vector<int32_t> int32_values(values.size());
            std::transform(values.begin(), values.end(), int32_values.begin(), [](int64_t v) {
                return static_cast<int32_t>(v);
            });
            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, int32_values)};
        }
        break;
    }
    case element::f32: {
        if (node.has_attribute("fp32_values")) {
            std::vector<float> values = node.get_attribute<std::vector<float>>("fp32_values");
            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        } else {
            auto values = node.get_attribute<std::vector<double>>("values");
            std::vector<float> values_f32(values.size());
            std::transform(values.begin(), values.end(), values_f32.begin(), [](double v) {
                return static_cast<float>(v);
            });
            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values_f32)};
        }
        break;
    }
    case element::f64: {
        auto values = node.get_attribute<std::vector<double>>("values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    case element::boolean: {
        auto values = node.get_attribute<std::vector<int32_t>>("bool_values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    case element::i64: {
        auto values = node.has_attribute("int64_values") ? node.get_attribute<std::vector<int64_t>>("int64_values")
                                                         : node.get_attribute<std::vector<int64_t>>("values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    default: {
        std::ostringstream oss;
        oss << "assign_value only supports int32, int64, float32, float64, bool, but receive dtype["
            << dtype.get_type_name() << "]";
        PADDLE_OP_CHECK(node, false, oss.str());
        break;
    }
    }

    return node.default_single_output_mapping({const_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
