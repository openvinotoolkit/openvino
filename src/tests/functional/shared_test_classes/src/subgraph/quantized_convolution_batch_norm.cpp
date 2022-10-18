// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_convolution_batch_norm.hpp"
#include "ngraph_functions/builders.hpp"
#include <exec_graph_info.hpp>

namespace SubgraphTestsDefinitions {

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

std::string QuantizedConvolutionBatchNorm::getTestCaseName(const testing::TestParamInfo<QuantizedConvolutionBatchNormParams>& obj) {
    std::ostringstream name;
    QuantizeType quantize_type;
    bool transpose_on_weights;
    std::string device;
    std::tie(quantize_type, transpose_on_weights, device) = obj.param;
    name << "quantize_type=" << quantize_type << "_transpose_on_weights=" << std::boolalpha << transpose_on_weights << "_device=" << device;
    return name.str();
}

using namespace ngraph;

void QuantizedConvolutionBatchNorm::SetUp() {
    QuantizeType quantize_type;
    bool transpose_on_weights;
    std::tie(quantize_type, transpose_on_weights, targetDevice) = GetParam();

    auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 16, 14, 14});
    auto low_act = opset8::Constant::create(element::f32, Shape{}, {0});
    auto high_act = opset8::Constant::create(element::f32, Shape{}, {20});
    std::shared_ptr<Node> activations = nullptr;
    auto low_weights = opset8::Constant::create(element::f32, Shape{}, {-0.72519057});
    auto high_weights = opset8::Constant::create(element::f32, Shape{}, {0.72519057});
    std::shared_ptr<Node> weights = nullptr;
    if (transpose_on_weights)
        weights = builder::makeConstant(element::f32, Shape{1, 16, 1, 16}, {}, true, 0.5f, -0.5f);
    else
        weights = builder::makeConstant(element::f32, Shape{16, 16, 1, 1}, {}, true, 0.5f, -0.5f);
    if (quantize_type == QuantizeType::FAKE_QUANTIZE) {
        activations = std::make_shared<opset8::FakeQuantize>(parameter, low_act, high_act, low_act, high_act, 256);
        weights = std::make_shared<opset8::FakeQuantize>(weights, low_weights, high_weights, low_weights, high_weights, 255);
    } else if (quantize_type == QuantizeType::QUANTIZE_DEQUANTIZE) {
        auto output_low_act = opset8::Constant::create(element::f32, Shape{}, {0});
        auto output_high_act = opset8::Constant::create(element::f32, Shape{}, {255});
        activations = std::make_shared<opset8::FakeQuantize>(parameter, low_act, high_act, output_low_act, output_high_act, 256);
        activations = std::make_shared<opset8::Convert>(activations, element::u8);
        activations = std::make_shared<opset8::Convert>(activations, element::f32);
        std::shared_ptr<Node> shift_act = opset8::Constant::create(element::u8, Shape{}, {0});
        shift_act = std::make_shared<opset8::Convert>(shift_act, element::f32);
        activations = std::make_shared<opset8::Subtract>(activations, shift_act);
        auto scale_act = opset8::Constant::create(element::f32, Shape{}, {20.0 / 255.0});
        activations = std::make_shared<opset8::Multiply>(activations, scale_act);

        auto output_low_weights = opset8::Constant::create(element::f32, Shape{}, {0});
        auto output_high_weights = opset8::Constant::create(element::f32, Shape{}, {254});
        weights = std::make_shared<opset8::FakeQuantize>(weights, low_weights, high_weights, output_low_weights, output_high_weights, 255);
        weights = std::make_shared<opset8::Convert>(weights, element::i8);
        weights = std::make_shared<opset8::Convert>(weights, element::f32);
        std::shared_ptr<Node> shift_weights = opset8::Constant::create(element::i8, Shape{}, {0});
        shift_weights = std::make_shared<opset8::Convert>(shift_weights, element::f32);
        weights = std::make_shared<opset8::Subtract>(weights, shift_weights);
        auto scale_weights = opset8::Constant::create(element::f32, Shape{}, {2.0 / 255.0});
        weights = std::make_shared<opset8::Multiply>(weights, scale_weights);
    } else if (quantize_type == QuantizeType::COMPRESSED_WEIGHTS) {
        auto output_low_act = opset8::Constant::create(element::f32, Shape{}, {0});
        auto output_high_act = opset8::Constant::create(element::f32, Shape{}, {255});
        activations = std::make_shared<opset8::FakeQuantize>(parameter, low_act, high_act, output_low_act, output_high_act, 256);
        activations = std::make_shared<opset8::Convert>(activations, element::u8);
        activations = std::make_shared<opset8::Convert>(activations, element::f32);
        std::shared_ptr<Node> shift_act = opset8::Constant::create(element::u8, Shape{}, {0});
        shift_act = std::make_shared<opset8::Convert>(shift_act, element::f32);
        activations = std::make_shared<opset8::Subtract>(activations, shift_act);
        auto scale_act = opset8::Constant::create(element::f32, Shape{}, {20.0 / 255.0});
        activations = std::make_shared<opset8::Multiply>(activations, scale_act);

        auto output_low_weights = opset8::Constant::create(element::f32, Shape{}, {0});
        auto output_high_weights = opset8::Constant::create(element::f32, Shape{}, {254});
        weights = std::make_shared<opset8::FakeQuantize>(weights, low_weights, high_weights, output_low_weights, output_high_weights, 255);
        weights = std::make_shared<opset8::Convert>(weights, element::i8);
        weights = get_constant_from_source(weights);
        weights = std::make_shared<opset8::Convert>(weights, element::f32);
        auto scale_weights = opset8::Constant::create(element::f32, Shape{}, {2.0 / 255.0});
        weights = std::make_shared<opset8::Multiply>(weights, scale_weights);
    }
    if (transpose_on_weights)
        weights = std::make_shared<opset8::Transpose>(weights, opset8::Constant::create(element::i32, Shape{4}, {1, 3, 0, 2}));
    auto conv = std::make_shared<opset8::Convolution>(activations, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
    auto gamma = builder::makeConstant(element::f32, Shape{16}, {}, true, 1.0f, 0.1f);
    auto beta = builder::makeConstant(element::f32, Shape{16}, {}, true, 1.0f, 0.1f);
    auto mean = builder::makeConstant(element::f32, Shape{16}, {}, true, 1.0f, 0.1f);
    auto var = builder::makeConstant(element::f32, Shape{16}, {}, true, 1.0f, 0.1f);
    auto batch_norm = std::make_shared<opset8::BatchNormInference>(conv, gamma, beta, mean, var, 0.00001);
    function = std::make_shared<ngraph::Function>(batch_norm, ParameterVector{parameter});
}

void QuantizedConvolutionBatchNorm::TearDown() {
    auto get_layer_type = [] (const std::shared_ptr<ngraph::Node>& node) -> const std::string& {
        const auto& rt_info = node->get_rt_info();
        auto it = rt_info.find(ExecGraphInfoSerialization::LAYER_TYPE);
        IE_ASSERT(it != rt_info.end());
        return it->second.as<std::string>();
    };

    auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& layer_type = get_layer_type(node);
        if (layer_type == "Convolution") {
            ops_found++;
            auto inputs = node->input_values();
            ASSERT_EQ(element::u8, inputs[0].get_element_type());
            ASSERT_EQ(element::i8, inputs[1].get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}

}  // namespace SubgraphTestsDefinitions
