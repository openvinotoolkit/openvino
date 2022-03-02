// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>
#include <map>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
    ngraph::helpers::ActivationTypes,   // Activation type
    std::vector<size_t>,                // Input Shape
    InferenceEngine::Precision,         // Network precision
    std::string,                        // Device name
    std::map<std::string, std::string>  // Configuration
> GnaPwlParams;

namespace LayerTestsDefinitions {

class GnaPwlTest : public testing::WithParamInterface<GnaPwlParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GnaPwlParams> obj) {
        ngraph::helpers::ActivationTypes activation;
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(activation, inputShape, netPrecision,  targetName, additional_config) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetName;
        result << "act=" << int(activation);
        for (auto const& configItem : additional_config) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        ngraph::helpers::ActivationTypes activation_type;
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(activation_type, inputShape, netPrecision,  targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input_params = ngraph::builder::makeParams(ngPrc, {inputShape});

        std::shared_ptr<ov::op::v0::Result> result;
        switch (activation_type) {
            case ngraph::helpers::ActivationTypes::Clamp:
            {
                auto clamp = std::make_shared<ngraph::opset8::Clamp>(input_params[0], -50, 50);
                result = std::make_shared<ngraph::opset8::Result>(clamp);
                break;
            }
            case ngraph::helpers::PReLu:
            {
                auto constant = ngraph::op::Constant::create(ngraph::element::Type_t::f32, {1}, {0.1});
                auto prelu = std::make_shared<ngraph::opset8::PRelu>(input_params[0], constant);
                result = std::make_shared<ngraph::opset8::Result>(prelu);
                break;
            }
            default:
            {
                auto act = ngraph::builder::makeActivation(input_params[0], ngPrc, activation_type, inputShape);
                result = std::make_shared<ngraph::opset8::Result>(act);
                break;
            }
        }
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    }
};

TEST_P(GnaPwlTest, CompareWithRefs) {
    Run();
};

std::vector<size_t> input_shapes = {1, 10};

std::vector<ngraph::helpers::ActivationTypes> activations = {
    ngraph::helpers::ActivationTypes::Tanh,
    ngraph::helpers::ActivationTypes::Sigmoid,
    ngraph::helpers::ActivationTypes::Clamp,
    ngraph::helpers::ActivationTypes::Relu,
    ngraph::helpers::ActivationTypes::PReLu
};

std::vector<InferenceEngine::Precision> net_precisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> additional_configuration = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
};

INSTANTIATE_TEST_SUITE_P(smoke_GnaPwl, GnaPwlTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(activations),
                                ::testing::Values(input_shapes),
                                ::testing::ValuesIn(net_precisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(additional_configuration)),
                        GnaPwlTest::getTestCaseName);

} // namespace LayerTestsDefinitions