// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

template <typename ActivationType>
ov::OutputVector qlinear_activation(const ov::frontend::onnx::Node& node, const ActivationType& activation_fn) {
    common::default_op_checks(node, 5);

    const auto inputs = node.get_ov_inputs();
    auto input_tensor = inputs[0];
    auto input_scale = inputs[1];
    auto input_zero_point =
        (inputs[2].get_shape().empty()) ? v0::Constant::create(input_tensor.get_element_type(), {}, {0}) : inputs[2];
    auto output_scale = inputs[3];
    auto output_zero_point =
        (inputs.size() > 4) ? inputs[4] : v0::Constant::create(input_tensor.get_element_type(), {}, {0});

    CHECK_VALID_NODE(node,
                     (input_tensor.get_element_type() == element::i8 || input_tensor.get_element_type() == element::u8),
                     "Input tensor must be either int8 or uint8. Got: ",
                     input_tensor.get_element_type());

    auto input_subtracted = std::make_shared<v1::Subtract>(input_tensor, input_zero_point);
    auto input_dequantized =
        std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(input_subtracted, input_scale.get_element_type()),
                                       input_scale);

    auto activation_result = activation_fn(input_dequantized);

    auto scaled_result_float = std::make_shared<v1::Divide>(activation_result, output_scale);
    auto quantized_result =
        std::make_shared<v1::Add>(std::make_shared<v0::Convert>(scaled_result_float, input_tensor.get_element_type()),
                                  output_zero_point);

    return ov::OutputVector{quantized_result};
}

ov::OutputVector qlinear_sigmoid(const ov::frontend::onnx::Node& node) {
    return qlinear_activation(node, [](const std::shared_ptr<ov::Node>& input_dequantized) {
        return std::make_shared<v0::Sigmoid>(input_dequantized);
    });
}

ov::OutputVector qlinear_leaky_relu(const ov::frontend::onnx::Node& node) {
    return qlinear_activation(node, [&](const std::shared_ptr<ov::Node>& input_dequantized) {
        auto alpha =
            v0::Constant::create(input_dequantized->get_element_type(), {}, {node.get_attribute_value<float>("alpha")});
        return std::make_shared<v1::Maximum>(input_dequantized,
                                             std::make_shared<v1::Multiply>(input_dequantized, alpha));
    });
}

ov::OutputVector qlinear_avg_pool(const ov::frontend::onnx::Node& node) {
    return qlinear_activation(node, [&](const std::shared_ptr<ov::Node>& input_dequantized) {
        const auto kernel_shape = node.get_attribute_value<std::vector<int64_t>>("kernel_shape");
        auto strides = node.get_attribute_value<std::vector<int64_t>>("strides", std::vector<int64_t>{1});
        auto pads = node.get_attribute_value<std::vector<int64_t>>("pads", std::vector<int64_t>{0});
        const auto ceil_mode = node.get_attribute_value<int64_t>("ceil_mode", 0);
        const auto count_include_pad = node.get_attribute_value<int64_t>("count_include_pad", 0);
        const auto auto_pad = node.get_attribute_value<std::string>("auto_pad", "NOTSET");

        const auto input_rank = input_dequantized->get_shape().size();
        const size_t num_spatial_dims = input_rank - 2;

        pads.resize(num_spatial_dims * 2, pads.size() == 1 ? pads[0] : 0);
        strides.resize(num_spatial_dims, strides.size() == 1 ? strides[0] : 1);

        auto avg_pool = std::make_shared<v1::AvgPool>(input_dequantized,
                                                      Strides(strides.begin(), strides.end()),
                                                      Shape(pads.begin(), pads.begin() + num_spatial_dims),
                                                      Shape(pads.begin() + num_spatial_dims, pads.end()),
                                                      Shape(kernel_shape.begin(), kernel_shape.end()),
                                                      count_include_pad == 0,
                                                      ceil_mode != 0 ? RoundingType::CEIL : RoundingType::FLOOR,
                                                      auto_pad == "SAME_UPPER"   ? PadType::SAME_UPPER
                                                      : auto_pad == "SAME_LOWER" ? PadType::SAME_LOWER
                                                                                 : PadType::EXPLICIT);

        return avg_pool;
    });
}

bool register_multiple_operators(void) {
    ONNX_OP_M("QLinearSigmoid", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_sigmoid, MICROSOFT_DOMAIN);
    ONNX_OP_M("QLinearLeakyRelu", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_leaky_relu, MICROSOFT_DOMAIN);
    ONNX_OP_M("QLinearAveragePool", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_avg_pool, MICROSOFT_DOMAIN);
    return true;
}

static bool registered = register_multiple_operators();

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
