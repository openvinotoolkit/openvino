// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "ngraph_conversion_tests/conv_bias_fusion.hpp"
#include <ngraph/variant.hpp>

namespace NGraphConversionTestsDefinitions {


std::string ConvBiasFusion::getTestCaseName(const testing::TestParamInfo<std::string> & obj) {
    return "Device=" + obj.param;
}

std::string ConvBiasFusion::getOutputName() const {
    if (this->GetParam() == CommonTestUtils::DEVICE_GPU)
        return "add_cldnn_output_postprocess";
    else
        return "add";
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
    auto function = net.getFunction();
    ASSERT_NE(nullptr, function);

    for (const auto & op : function->get_ops()) {
        if (op->get_friendly_name() ==  getOutputName()) {
            auto rtInfo = op->get_rt_info();
            auto it = rtInfo.find("originalLayersNames");
            ASSERT_NE(rtInfo.end(), it);
            auto variant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            ASSERT_NE(nullptr, variant);
            ASSERT_EQ(variant->get(), "add,conv");
            break;
        }
    }
}

}  // namespace NGraphConversionTestsDefinitions