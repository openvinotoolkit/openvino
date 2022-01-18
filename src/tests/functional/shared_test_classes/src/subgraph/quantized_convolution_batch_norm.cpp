// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_convolution_batch_norm.hpp"
#include "ngraph_functions/builders.hpp"
#include <exec_graph_info.hpp>

namespace SubgraphTestsDefinitions {

std::string QuantizedConvolutionBatchNorm::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    return "device=" + obj.param;
}

using namespace ngraph;

void QuantizedConvolutionBatchNorm::SetUp() {
    targetDevice = GetParam();

    auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 16, 14, 14});
    auto low_act = opset8::Constant::create(element::f32, Shape{}, {0});
    auto high_act = opset8::Constant::create(element::f32, Shape{}, {20});
    std::shared_ptr<Node> activations = std::make_shared<opset8::FakeQuantize>(parameter, low_act, high_act, low_act, high_act, 256);
    auto low_weights = opset8::Constant::create(element::f32, Shape{}, {-0.72519057});
    auto high_weights = opset8::Constant::create(element::f32, Shape{}, {0.72519057});
    std::shared_ptr<Node> weights = builder::makeConstant(element::f32, Shape{16, 16, 1, 1}, {}, true, 0.5f, -0.5f);
    weights = std::make_shared<opset8::FakeQuantize>(weights, low_weights, high_weights, low_weights, high_weights, 255);
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
