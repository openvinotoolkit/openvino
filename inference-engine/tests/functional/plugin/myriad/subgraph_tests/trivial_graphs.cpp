// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>

#include <ngraph/opsets/opset6.hpp>
#include <ie_ngraph_utils.hpp>

#include "vpu/private_plugin_config.hpp"

namespace {

class ParameterResultTest : public testing::WithParamInterface<std::tuple<ngraph::Shape, ngraph::element::Type, std::string>>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& inShape = std::get<0>(parameters);
        const auto& inType = std::get<1>(parameters);
        targetDevice = std::get<2>(parameters);
        inPrc = outPrc = InferenceEngine::details::convertPrecision(inType);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        const auto param = std::make_shared<ngraph::opset6::Parameter>(inType, inShape);

        function = std::make_shared<ngraph::Function>(param->outputs(), ngraph::ParameterVector{param}, "ParameterResult");
    }
};

TEST_P(ParameterResultTest, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_TrivialGraph, ParameterResultTest,
                        ::testing::Combine(
                                ::testing::Values(
                                        ngraph::Shape{8, 800},
                                        ngraph::Shape{10, 1000}),
                                ::testing::Values(
                                        ngraph::element::f16,
                                        ngraph::element::f32),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

class ParameterShapeOfResultTest : public testing::WithParamInterface<std::tuple<ngraph::Shape, ngraph::element::Type, std::string>>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& inShape = std::get<0>(parameters);
        const auto& inType = std::get<1>(parameters);
        targetDevice = std::get<2>(parameters);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        const auto param = std::make_shared<ngraph::opset6::Parameter>(inType, inShape);
        param->set_friendly_name("parameter");
        const auto shapeOf = std::make_shared<ngraph::opset6::ShapeOf>(param, ngraph::element::i32);

        function = std::make_shared<ngraph::Function>(shapeOf->outputs(), ngraph::ParameterVector{param}, "ParameterShapeOfResult");
    }

    void ConfigureNetwork() override {
        const auto& inType = std::get<1>(GetParam());

        for (const auto& input : cnnNetwork.getInputsInfo()) {
            const auto& paramIt = input.second->name().find("parameter");
            if (paramIt != std::string::npos) {
                input.second->setPrecision(InferenceEngine::details::convertPrecision(inType));
            }
        }
    }
};

TEST_P(ParameterShapeOfResultTest, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_TrivialGraph, ParameterShapeOfResultTest,
                        ::testing::Combine(
                                ::testing::Values(
                                        ngraph::Shape{8, 800},
                                        ngraph::Shape{10, 1000}),
                                ::testing::Values(
                                        ngraph::element::f16,
                                        ngraph::element::f32),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

} // namespace
