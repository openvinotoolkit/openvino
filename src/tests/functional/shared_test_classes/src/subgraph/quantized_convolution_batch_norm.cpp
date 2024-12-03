// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/subgraph/quantized_convolution_batch_norm.hpp"

namespace ov {
namespace test {

std::ostream& operator<<(std::ostream& os, ConvType type) {
    switch (type) {
        case ConvType::CONVOLUTION:
            os << "convolution";
            break;
        case ConvType::CONVOLUTION_BACKPROP:
            os << "convolution_backprop";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, QuantizeType type) {
    switch (type) {
        case QuantizeType::FAKE_QUANTIZE:
            os << "fake_quantize";
            break;
        case QuantizeType::QUANTIZE_DEQUANTIZE:
            os << "quantize_dequantize";
            break;
        case QuantizeType::COMPRESSED_WEIGHTS:
            os << "compressed_weights";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, IntervalsType type) {
    switch (type) {
        case IntervalsType::PER_TENSOR:
            os << "per_tensor";
            break;
        case IntervalsType::PER_CHANNEL:
            os << "per_channel";
            break;
    }
    return os;
}

std::string QuantizedConvolutionBatchNorm::getTestCaseName(const testing::TestParamInfo<QuantizedConvolutionBatchNormParams>& obj) {
    std::ostringstream name;
    ConvType conv_type;
    QuantizeType quantize_type;
    IntervalsType intervals_type;
    bool transpose_on_weights;
    std::string device;
    std::tie(conv_type, quantize_type, intervals_type, transpose_on_weights, device) = obj.param;
    name << "conv_type=" << conv_type << "_quantize_type=" << quantize_type << "_intervals_type=" << intervals_type <<
        "_transpose_on_weights=" << std::boolalpha << transpose_on_weights << "_device=" << device;
    return name.str();
}

void QuantizedConvolutionBatchNorm::SetUp() {
    ConvType conv_type;
    QuantizeType quantize_type;
    IntervalsType intervals_type;
    bool transpose_on_weights;
    std::tie(conv_type, quantize_type, intervals_type, transpose_on_weights, targetDevice) = GetParam();

    size_t input_channels = 3;
    size_t output_channels = 4;
    auto parameter = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, input_channels, 14, 14});
    Shape input_intervals_shape;
    Shape weights_intervals_shape;
    Shape weights_shape{output_channels, input_channels, 1, 1};
    if (intervals_type == IntervalsType::PER_CHANNEL) {
        input_intervals_shape = {1, input_channels, 1, 1};
        weights_intervals_shape = {output_channels, 1, 1, 1};
    }
    if (conv_type == ConvType::CONVOLUTION_BACKPROP) {
        // while weights for Convolution have shape {output channels, input channels, ...},
        // weights for ConvolutionBackpropData have following shape {input channels, output channels, ...}
        std::swap(weights_shape[0], weights_shape[1]);
        if (weights_intervals_shape.size() == 4) {
            std::swap(weights_intervals_shape[0], weights_intervals_shape[1]);
        }
    }
    if (transpose_on_weights) {
        weights_shape = {weights_shape[2], weights_shape[0], weights_shape[3], weights_shape[1]};
        if (weights_intervals_shape.size() == 4) {
            weights_intervals_shape = {weights_intervals_shape[2], weights_intervals_shape[0], weights_intervals_shape[3], weights_intervals_shape[1]};
        }
    }
    auto low_act = ov::op::v0::Constant::create(element::f32, input_intervals_shape, {0});
    auto high_act = ov::op::v0::Constant::create(element::f32, input_intervals_shape, {20});
    auto low_weights = ov::op::v0::Constant::create(element::f32, weights_intervals_shape, {-0.72519057});
    auto high_weights = ov::op::v0::Constant::create(element::f32, weights_intervals_shape, {0.72519057});
    std::shared_ptr<Node> activations = nullptr;
    auto weights_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(element::f32, weights_shape, -0.5f, 0.5f, 1);
    std::shared_ptr<Node> weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    if (quantize_type == QuantizeType::FAKE_QUANTIZE) {
        activations = std::make_shared< ov::op::v0::FakeQuantize>(parameter, low_act, high_act, low_act, high_act, 256);
        weights = std::make_shared< ov::op::v0::FakeQuantize>(weights, low_weights, high_weights, low_weights, high_weights, 255);
    } else if (quantize_type == QuantizeType::QUANTIZE_DEQUANTIZE) {
        auto output_low_act =  ov::op::v0::Constant::create(element::f32, Shape{}, {0});
        auto output_high_act =  ov::op::v0::Constant::create(element::f32, Shape{}, {255});
        activations = std::make_shared< ov::op::v0::FakeQuantize>(parameter, low_act, high_act, output_low_act, output_high_act, 256);
        activations = std::make_shared< ov::op::v0::Convert>(activations, element::u8);
        activations = std::make_shared< ov::op::v0::Convert>(activations, element::f32);
        std::shared_ptr<Node> shift_act =  ov::op::v0::Constant::create(element::u8, input_intervals_shape, {0});
        shift_act = std::make_shared< ov::op::v0::Convert>(shift_act, element::f32);
        activations = std::make_shared<ov::op::v1::Subtract>(activations, shift_act);
        auto scale_act = ov::op::v0::Constant::create(element::f32, input_intervals_shape, {20.0 / 255.0});
        activations = std::make_shared<ov::op::v1::Multiply>(activations, scale_act);

        auto output_low_weights = ov::op::v0::Constant::create(element::f32, Shape{}, {0});
        auto output_high_weights = ov::op::v0::Constant::create(element::f32, Shape{}, {254});
        weights = std::make_shared<ov::op::v0::FakeQuantize>(weights, low_weights, high_weights, output_low_weights, output_high_weights, 255);
        weights = std::make_shared<ov::op::v0::Convert>(weights, element::i8);
        weights = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
        std::shared_ptr<Node> shift_weights = ov::op::v0::Constant::create(element::i8, weights_intervals_shape, {0});
        shift_weights = std::make_shared<ov::op::v0::Convert>(shift_weights, element::f32);
        weights = std::make_shared<ov::op::v1::Subtract>(weights, shift_weights);
        auto scale_weights = ov::op::v0::Constant::create(element::f32, weights_intervals_shape, {2.0 / 255.0});
        weights = std::make_shared<ov::op::v1::Multiply>(weights, scale_weights);
    } else if (quantize_type == QuantizeType::COMPRESSED_WEIGHTS) {
        auto output_low_act = ov::op::v0::Constant::create(element::f32, Shape{}, {0});
        auto output_high_act = ov::op::v0::Constant::create(element::f32, Shape{}, {255});
        activations = std::make_shared<ov::op::v0::FakeQuantize>(parameter, low_act, high_act, output_low_act, output_high_act, 256);
        activations = std::make_shared<ov::op::v0::Convert>(activations, element::u8);
        activations = std::make_shared<ov::op::v0::Convert>(activations, element::f32);
        std::shared_ptr<Node> shift_act = ov::op::v0::Constant::create(element::u8, input_intervals_shape, {0});
        shift_act = std::make_shared<ov::op::v0::Convert>(shift_act, element::f32);
        activations = std::make_shared<ov::op::v1::Subtract>(activations, shift_act);
        auto scale_act = ov::op::v0::Constant::create(element::f32, input_intervals_shape, {20.0 / 255.0});
        activations = std::make_shared<ov::op::v1::Multiply>(activations, scale_act);

        auto output_low_weights = ov::op::v0::Constant::create(element::f32, Shape{}, {0});
        auto output_high_weights = ov::op::v0::Constant::create(element::f32, Shape{}, {254});
        weights = std::make_shared<ov::op::v0::FakeQuantize>(weights, low_weights, high_weights, output_low_weights, output_high_weights, 255);
        weights = std::make_shared<ov::op::v0::Convert>(weights, element::i8);
        weights = ov::util::get_constant_from_source(weights);
        weights = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
        auto scale_weights = ov::op::v0::Constant::create(element::f32, weights_intervals_shape, {2.0 / 255.0});
        weights = std::make_shared<ov::op::v1::Multiply>(weights, scale_weights);
    }
    if (transpose_on_weights) {
        weights = std::make_shared<ov::op::v1::Transpose>(weights, ov::op::v0::Constant::create(element::i32, Shape{4}, {1, 3, 0, 2}));
    }
    std::shared_ptr<Node> conv;
    if (conv_type == ConvType::CONVOLUTION) {
        conv = std::make_shared<ov::op::v1::Convolution>(activations, weights, Strides{1, 1},
                CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
    } else {
        conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(activations, weights, Strides{1, 1},
                CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
    }

    auto gamma_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(element::f32, Shape{output_channels}, 0.1f, 1.f, 1);
    auto gamma = std::make_shared<ov::op::v0::Constant>(gamma_tensor);
    auto beta_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(element::f32, Shape{output_channels}, 0.1f, 1.f, 1);
    auto beta = std::make_shared<ov::op::v0::Constant>(beta_tensor);
    auto mean_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(element::f32, Shape{output_channels}, 0.1f, 1.f, 1);
    auto mean = std::make_shared<ov::op::v0::Constant>(mean_tensor);
    auto var_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(element::f32, Shape{output_channels}, 0.1f, 1.f, 1);
    auto var = std::make_shared<ov::op::v0::Constant>(var_tensor);
    auto batch_norm = std::make_shared<ov::op::v5::BatchNormInference>(conv, gamma, beta, mean, var, 0.00001);
    function = std::make_shared<ov::Model>(batch_norm, ParameterVector{parameter});
}

void QuantizedConvolutionBatchNorm::TearDown() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto get_layer_type = [] (const std::shared_ptr<ov::Node>& node) -> const std::string& {
        const auto& rt_info = node->get_rt_info();
        auto it = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        OPENVINO_ASSERT(it != rt_info.end());
        return it->second.as<std::string>();
    };

    auto runtime_function = compiledModel.get_runtime_model();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& layer_type = get_layer_type(node);
        if (layer_type == "Convolution" || layer_type == "Deconvolution") {
            ops_found++;
            auto inputs = node->input_values();
            ASSERT_EQ(element::u8, inputs[0].get_element_type());
            ASSERT_EQ(element::i8, inputs[1].get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}

}  // namespace test
}  // namespace ov
