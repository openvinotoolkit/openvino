// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_conversion_tests/plugin_specific_ngraph_conversion.hpp"

namespace NGraphConversionTestsDefinitions {

void PluginSpecificConversion::SetUp() {
    device = this->GetParam();
}

std::string PluginSpecificConversion::getTestCaseName(const testing::TestParamInfo<std::string> & obj) {
    return obj.param;
}

TEST_P(PluginSpecificConversion, GeluConversionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
        auto gelu = std::make_shared<ngraph::op::Gelu>(input);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input});
    }

    auto network = InferenceEngine::CNNNetwork(f);

    InferenceEngine::Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device);
    auto net = exeNetwork.GetExecGraphInfo();

    if (device == "CPU") {
        // Parameter->Activation->Output
        ASSERT_EQ(net.layerCount(), 3);
    } else if (device == "GPU") {
        // Parameter--->ScaleShift-------------->Eltwise-->Result
        //          `-->ScaleShift->ScaleShift-`
        ASSERT_EQ(net.layerCount(), 6);
    }
}

TEST_P(PluginSpecificConversion, MatMulConversionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{64, 3}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input, weights);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input});
    }

    auto network = InferenceEngine::CNNNetwork(f);

    InferenceEngine::Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device);
    auto net = exeNetwork.GetExecGraphInfo();

    // TODO: this test is in progress and will be finished when 3D FC will be supported
}
}  // namespace NGraphConversionTestsDefinitions
