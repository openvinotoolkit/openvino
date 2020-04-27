// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_conversion_tests/conv_bias_fusion.hpp"

namespace NGraphConversionTestsDefinitions {


std::string ConvBiasFusion::getTestCaseName(const testing::TestParamInfo<std::string> & obj) {
    return "Device=" + obj.param;
}

TEST_P(ConvBiasFusion, ConvBiasFusion) {
    std::string device = this->GetParam();
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{6, 3, 1, 1}, {1});
        auto biases = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{6, 1, 1}, {1});
        auto conv = std::make_shared<ngraph::opset1::Convolution>(input, weights, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{1, 1});
        auto add = std::make_shared<ngraph::opset1::Add>(conv, biases);

        input->set_friendly_name("parameter");
        conv->set_friendly_name("conv");
        add->set_friendly_name("add");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    }

    auto network = InferenceEngine::CNNNetwork(f);

    InferenceEngine::Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device);
    auto net = exeNetwork.GetExecGraphInfo();

    IE_SUPPRESS_DEPRECATED_START
    auto add_layer = net.getLayerByName("add");
    ASSERT_EQ(add_layer->params["originalLayersNames"], "add,conv");
    IE_SUPPRESS_DEPRECATED_END
}

}  // namespace NGraphConversionTestsDefinitions